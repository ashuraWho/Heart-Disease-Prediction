# ============================================================ #
# Module 04 – Inference (Tournament Optimized)                 #
# ============================================================ #

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

try:
    from shared_utils import (
        setup_environment, 
        console, 
        ARTIFACTS_DIR, 
        CLINICAL_GUIDE, 
        init_db, 
        get_db_connection
    )
except ImportError:
    print("Error: shared_utils not found.")
    sys.exit(1)

setup_environment()

import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
from catboost import CatBoostClassifier
from rich.prompt import Prompt, FloatPrompt
from rich.panel import Panel

# --- LOAD ENSEMBLE ---
def load_resources():
    try:
        preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib")
        
        models = {}
        models["LR"] = load(ARTIFACTS_DIR / "model_lr.joblib")
        models["RF"] = load(ARTIFACTS_DIR / "model_rf.joblib")
        models["ET"] = load(ARTIFACTS_DIR / "model_et.joblib")
        models["GB"] = load(ARTIFACTS_DIR / "model_gb.joblib")
        models["XGB"] = load(ARTIFACTS_DIR / "model_xgb.joblib")
        models["LGBM"] = load(ARTIFACTS_DIR / "model_lgbm.joblib")
        
        # Load CatBoost specifically
        cat = CatBoostClassifier()
        cat.load_model(str(ARTIFACTS_DIR / "model_cat.cbm"))
        models["CAT"] = cat
        
        models["DL"] = tf.keras.models.load_model(ARTIFACTS_DIR / "model_dl.keras")
            
        threshold = 0.5
        thresh_path = ARTIFACTS_DIR / "threshold.txt"
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                threshold = float(f.read().strip())
        
        return preprocessor, models, threshold
    except Exception as e:
        console.print(f"[bold red]Load Error: {e}[/bold red]")
        sys.exit(1)

# --- DB HELPERS ---
def save_to_db(data, prediction, probability):
    with get_db_connection() as conn:
        df_to_save = data.copy()
        df_to_save['Prediction'] = int(prediction)
        df_to_save['Probability'] = float(probability)
        try:
             df_to_save.to_sql('patients', conn, if_exists='append', index=False)
        except Exception:
            pass

# --- FEATURE ENGINEERING ---
def add_engineered_features(df):
    gen_health_map = {"Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5}
    if 'GeneralHealth' in df.columns:
        df['GeneralHealth_Num'] = df['GeneralHealth'].map(gen_health_map).fillna(3)

    if 'SleepHours' in df.columns and 'PhysicalHealthDays' in df.columns:
        df['Sleep_Health_Ratio'] = df['SleepHours'] / (df['PhysicalHealthDays'] + 1)
        
    return df

# --- INTERACTIVE INPUT ---
def get_interactive_input():
    console.print(Panel("[bold]Enter Patient Clinical Data (2022 Standard)[/bold]", style="cyan"))
    data = {}
    
    # Demographics
    data["Sex"] = Prompt.ask("[green]Sex[/green]", choices=["Male", "Female"])
    data["AgeCategory"] = Prompt.ask("[green]Age Category[/green]", choices=[
        "Age 18-24", "Age 25-29", "Age 30-34", "Age 35-39", "Age 40-44", "Age 45-49", 
        "Age 50-54", "Age 55-59", "Age 60-64", "Age 65-69", "Age 70-74", "Age 75-79", "Age 80 or older"
    ], default="Age 60-64")
    data["RaceEthnicityCategory"] = Prompt.ask("[green]Race/Ethnicity[/green]", choices=[
        "White only, Non-Hispanic", "Black only, Non-Hispanic", "Hispanic", "Other", "Multiracial, Non-Hispanic", "Asian only, Non-Hispanic"
    ], default="White only, Non-Hispanic")
    data["HeightInMeters"] = FloatPrompt.ask("[green]Height (m)[/green] e.g. 1.75")
    data["WeightInKilograms"] = FloatPrompt.ask("[green]Weight (kg)[/green] e.g. 80.0")
    data["BMI"] = data["WeightInKilograms"] / (data["HeightInMeters"] ** 2)
    
    # Health
    data["GeneralHealth"] = Prompt.ask("[green]General Health[/green]", choices=["Excellent", "Very good", "Good", "Fair", "Poor"])
    data["PhysicalHealthDays"] = FloatPrompt.ask("[green]Physical Health Days[/green]", default=0.0)
    data["MentalHealthDays"] = FloatPrompt.ask("[green]Mental Health Days[/green]", default=0.0)
    data["SleepHours"] = FloatPrompt.ask("[green]Sleep Hours[/green]", default=7.0)
    
    # Habits
    data["PhysicalActivities"] = Prompt.ask("[green]Physical Activities[/green]", choices=["Yes", "No"])
    data["SmokerStatus"] = Prompt.ask("[green]Smoking Status[/green]", choices=[
        "Never smoked", "Former smoker", "Current smoker - now smokes some days", "Current smoker - now smokes every day"
    ])
    data["ECigaretteUsage"] = Prompt.ask("[green]E-Cigarette Usage[/green]", choices=[
        "Never used e-cigarettes in my entire life", "Not at all (right now)", "Use them some days", "Use them every day"
    ], default="Never used e-cigarettes in my entire life")
    data["AlcoholDrinkers"] = Prompt.ask("[green]Alcohol Drinkers[/green]", choices=["Yes", "No"])

    # Conditions
    conditions = [
        "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD", 
        "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes",
        "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
        "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands",
        "ChestScan"
    ]
    console.print("[dim]Answer Yes/No for condition history:[/dim]")
    for cond in conditions:
        clean_cond = cond.replace("Had", "").replace("Difficulty", "Diff ").replace("Or", "/")
        data[cond] = Prompt.ask(f"[green]{clean_cond}[/green]?", choices=["Yes", "No"], default="No")

    return pd.DataFrame([data])

# --- PREDICTION ---
def predict_ensemble(preprocessor, models, threshold):
    raw_df = get_interactive_input()
    df = add_engineered_features(raw_df)
    
    try:
        X = preprocessor.transform(df)
        
        # 1. Base Predictions
        p_lr = models["LR"].predict_proba(X)[0][1]
        p_rf = models["RF"].predict_proba(X)[0][1]
        p_et = models["ET"].predict_proba(X)[0][1]
        p_gb = models["GB"].predict_proba(X)[0][1]
        p_xgb = models["XGB"].predict_proba(X)[0][1]
        p_lgbm = models["LGBM"].predict_proba(X)[0][1]
        p_cat = models["CAT"].predict_proba(X)[0][1]
        p_dl = models["DL"].predict(X, verbose=0)[0][0]
        
        # 2. Weighted Average
        # XGB (4), LGBM (4), CAT (4), ET (3), RF (2), GB (2), DL (2), LR (0)
        weights = {
            "LR": 0, "RF": 2, "ET": 3, "GB": 2, 
            "XGB": 4, "LGBM": 4, "CAT": 4, "DL": 2
        }
        total_w = sum(weights.values())
        
        avg_prob = (
            p_lr*0 + p_rf*2 + p_et*3 + p_gb*2 + 
            p_xgb*4 + p_lgbm*4 + p_cat*4 + p_dl*2
        ) / total_w

        pred = 1 if avg_prob >= threshold else 0
        
        style = "bold red" if pred == 1 else "bold green"
        msg = "High Risk (Detection)" if pred == 1 else "Low Risk (Healthy)"
        
        console.print(Panel(
            f"[bold]Mega-Ensemble Prediction:[/bold] [{style}]{msg}[/]\n"
            f"[bold]Confidence:[/bold] {avg_prob:.2%}\n"
            f"[dim]Threshold (Tournament Best): {threshold:.2%}[/dim]\n"
            f"[dim]Models: XGB, LGBM, CAT, ET, RF, GB, DL[/dim]",
            title="Optimized Result",
            border_style=style
        ))
        
        save_to_db(raw_df, pred, avg_prob)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

# --- MAIN ---
if __name__ == "__main__":
    init_db()
    preprocessor, models, threshold = load_resources()
    
    if "--history" in sys.argv:
        pass
    else:
        predict_ensemble(preprocessor, models, threshold)
