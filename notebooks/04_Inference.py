# ============================================================ #
# Module 04 – Inference, Database & Clinical Support           #
# ============================================================ #

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from shared_utils import (
        setup_environment, 
        console, 
        ARTIFACTS_DIR, 
        CLINICAL_GUIDE, 
        MAPPINGS, 
        init_db, 
        get_db_connection
    )
except ImportError:
    print("Error: shared_utils not found.")
    sys.exit(1)

# Initialize Environment
setup_environment()

import pandas as pd
import numpy as np
import sqlite3
from joblib import load
import tensorflow as tf
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.table import Table
from rich.panel import Panel

# --- LOAD RESOURCES ---
def load_resources():
    try:
        preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib")
        with open(ARTIFACTS_DIR / "model_type.txt", "r") as f: 
            model_type = f.read().strip()
            
        if model_type == "keras": 
            model = tf.keras.models.load_model(ARTIFACTS_DIR / "best_model_unified.keras")
        else: 
            model = load(ARTIFACTS_DIR / "best_model_unified.joblib")
            
        return preprocessor, model, model_type
    except Exception as e:
        console.print(f"[bold red]Load Error: {e}[/bold red]")
        console.print("[yellow]Please run Module 01 & 02 first.[/yellow]")
        sys.exit(1)

# --- DB HELPERS ---
def save_to_db(data, prediction, probability):
    with get_db_connection() as conn:
        df_to_save = data.copy()
        df_to_save['Prediction'] = int(prediction)
        df_to_save['Probability'] = float(probability)
        df_to_save.to_sql('patients', conn, if_exists='append', index=False)

# --- INTERACTIVE INPUT ---
def get_interactive_input():
    console.print(Panel("[bold]Enter Patient Clinical Data[/bold]", style="cyan"))
    
    raw_data = {}
    numeric_data = {}
    
    # Define numeric fields
    numeric_fields = ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", 
                      "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]
    
    # 1. Numeric Inputs
    for key in numeric_fields:
        val = FloatPrompt.ask(f"[green]{key}[/green] ({CLINICAL_GUIDE.get(key, '')})")
        raw_data[key] = [val]
        numeric_data[key] = [val]
        
    # 2. Gender
    gender = Prompt.ask("[green]Gender[/green]", choices=["Male", "Female"])
    raw_data["Gender"] = [gender]
    numeric_data["Gender"] = [MAPPINGS["Binary"][gender]]
    
    # 3. Binary Questions
    binary_questions = [
        "Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure", 
        "Low HDL Cholesterol", "High LDL Cholesterol"
    ]
    for key in binary_questions:
        val = Prompt.ask(f"[green]{key}[/green]?", choices=["Yes", "No"])
        raw_data[key] = [val]
        numeric_data[key] = [MAPPINGS["Binary"][val]]
        
    # 4. Ordinal inputs
    ordinal_basics = ["Exercise Habits", "Stress Level", "Sugar Consumption"]
    for key in ordinal_basics:
        val = Prompt.ask(f"[green]{key}[/green]", choices=["Low", "Medium", "High"])
        raw_data[key] = [val]
        numeric_data[key] = [MAPPINGS["Ordinal_Basic"][val]]
        
    # 5. Alcohol
    alc = Prompt.ask("[green]Alcohol Consumption[/green]", choices=["None", "Low", "Medium", "High"])
    raw_data["Alcohol Consumption"] = [alc]
    numeric_data["Alcohol Consumption"] = [MAPPINGS["Ordinal_Alcohol"][alc]]
    
    return pd.DataFrame(raw_data), pd.DataFrame(numeric_data)

# --- PREDICTION LOGIC ---
def predict_single(preprocessor, model, model_type):
    raw_df, numeric_df = get_interactive_input()
    
    try:
        processed = preprocessor.transform(numeric_df)
        
        if model_type == "keras":
            prob = model.predict(processed, verbose=0)[0][0]
            pred = 1 if prob >= 0.5 else 0
        else:
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else 0.0
            
        # Display Result
        style = "bold red" if pred == 1 else "bold green"
        msg = "Presence Detected (High Risk)" if pred == 1 else "No Disease Detected (Low Risk)"
        
        console.print(Panel(
            f"[bold]Prediction:[/bold] [{style}]{msg}[/{style}]\n"
            f"[bold]Probability:[/bold] {prob:.2%}",
            title="Inference Result",
            border_style=style
        ))
        
        save_to_db(raw_df, pred, prob)
        console.print("[dim]Patient record saved to database.[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Prediction Error: {e}[/bold red]")

def batch_predict_from_db(preprocessor, model, model_type):
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM patients", conn)
            if df.empty:
                console.print("[yellow]Database is empty.[/yellow]")
                return
                
            # Prepare numeric DF
            df_numeric = df.copy().drop(columns=['id', 'Prediction', 'Probability', 'Timestamp'], errors='ignore')
            
            # Map columns back to numbers (similar to mapping logic)
            # Note: This relies on the raw strings stored in DB being mappable.
            # Ideally, we should store numeric codes or handle this robustly.
            # Re-implementing mapping for consistency with raw inputs:
            
            for col in df_numeric.columns:
                if col in ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", 
                           "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                
                elif col in ["Gender", "Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure", 
                           "Low HDL Cholesterol", "High LDL Cholesterol"]:
                    df_numeric[col] = df_numeric[col].map(MAPPINGS["Binary"])
                    
                elif col == "Alcohol Consumption":
                    df_numeric[col] = df_numeric[col].map(MAPPINGS["Ordinal_Alcohol"])
                    
                elif col in ["Exercise Habits", "Stress Level", "Sugar Consumption"]:
                    df_numeric[col] = df_numeric[col].map(MAPPINGS["Ordinal_Basic"])

            # Handle any conversion failures (fill with mean or mode if needed, here just drop/clean)
            df_numeric.fillna(0, inplace=True) # Simple fallback
            
            processed = preprocessor.transform(df_numeric)
            
            if model_type == "keras":
                df['New_Prob'] = model.predict(processed, verbose=0).ravel()
                df['New_Pred'] = (df['New_Prob'] >= 0.5).astype(int)
            else:
                df['New_Pred'] = model.predict(processed)

            # Show Table
            table = Table(title="Batch Prediction History")
            table.add_column("ID", justify="right", style="cyan")
            table.add_column("Age", justify="right")
            table.add_column("Gender")
            table.add_column("Hist Pred", justify="center")
            table.add_column("New Pred", justify="center", style="bold")
            
            for index, row in df.iterrows():
                table.add_row(
                    str(row['id']), 
                    str(row['Age']), 
                    str(row['Gender']), 
                    str(row['Prediction']),
                    f"[red]1[/red]" if row['New_Pred'] == 1 else "[green]0[/green]"
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[bold red]Batch Error: {e}[/bold red]")

# --- MAIN ---
if __name__ == "__main__":
    init_db() # Ensure DB exists
    preprocessor, model, model_type = load_resources()
    
    if "--batch" in sys.argv:
        batch_predict_from_db(preprocessor, model, model_type)
    else:
        predict_single(preprocessor, model, model_type)
