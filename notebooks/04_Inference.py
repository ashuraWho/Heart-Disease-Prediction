# ============================================================ #
# Module 04 – Inference (Tournament Optimized)                 #
# ============================================================ #
# Questo modulo gestisce:
# - caricamento dei modelli addestrati
# - input clinico interattivo
# - feature engineering coerente con il training
# - inferenza tramite mega-ensemble pesato
# - salvataggio dei risultati su database
# È pensato come ENTRY POINT finale del progetto.
# ============================================================ #

import sys
from pathlib import Path

# Aggiunge la directory del modulo al PYTHONPATH
# Serve per rendere importabile shared_utils anche se il progetto
# viene eseguito da una posizione arbitraria
sys.path.append(str(Path(__file__).resolve().parent))

# Import centralizzato delle utility condivise
try:
    from shared_utils import (
        setup_environment, # Setup globale (log, rich, seed, dirs)
        console, # Console Rich per output stilizzato
        ARTIFACTS_DIR, # Directory con modelli e preprocessori
        CLINICAL_GUIDE, # (Opzionale) guida clinica testuale
        init_db, # Inizializzazione database SQLite
        get_db_connection # Connessione DB contestuale
    )
except ImportError:
    # Fallback esplicito se la struttura del progetto è corrotta
    print("Error: shared_utils not found.")
    sys.exit(1)

# Inizializza ambiente (stile, seed, directories)
setup_environment()

# --- LIBRERIE CORE ---
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
from catboost import CatBoostClassifier
from rich.prompt import Prompt, FloatPrompt
from rich.panel import Panel


# ============================================================ #
# LOAD MODELS & PREPROCESSOR
# ============================================================ #
# Carica:
# - preprocessor (ColumnTransformer)
# - tutti i modelli dell'ensemble
# - soglia ottimizzata (tournament best)
# ============================================================ #

def load_resources():
    """
    Carica tutti i modelli addestrati e il preprocessor necessario per l'inferenza.
    
    Questa funzione carica:
    - Preprocessor (ColumnTransformer): Pipeline di preprocessing identica al training
    - 7 modelli ML classici: Logistic Regression, Random Forest, Extra Trees,
      Gradient Boosting, XGBoost, LightGBM
    - Modello CatBoost: Formato proprietario .cbm
    - Modello Deep Learning: TensorFlow/Keras formato .keras
    - Threshold ottimizzato: Soglia decisionale dal Threshold Tournament
    
    Returns:
        tuple: (preprocessor, models_dict, threshold) dove:
            - preprocessor: ColumnTransformer fittato (StandardScaler + OneHotEncoder)
            - models_dict: Dizionario con chiavi ["LR", "RF", "ET", "GB", "XGB", "LGBM", "CAT", "DL"]
            - threshold: float, soglia decisionale ottimizzata (default 0.5 se non trovata)
    
    Raises:
        SystemExit: Se il caricamento fallisce (modelli o preprocessor mancanti)
    """
    try:
        # ============================================================
        # CARICAMENTO PREPROCESSOR
        # ============================================================
        # Il preprocessor contiene la pipeline completa di preprocessing:
        # - StandardScaler per feature numeriche
        # - OneHotEncoder per feature categoriche
        # Deve essere identico a quello usato durante il training
        # ============================================================
        preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib")
        
        # ============================================================
        # CARICAMENTO MODELLI ML CLASSICI (formato joblib)
        # ============================================================
        # Modelli scikit-learn e boosting che usano joblib per serializzazione
        # ============================================================
        models = {}
        models["LR"] = load(ARTIFACTS_DIR / "model_lr.joblib")      # Logistic Regression
        models["RF"] = load(ARTIFACTS_DIR / "model_rf.joblib")      # Random Forest
        models["ET"] = load(ARTIFACTS_DIR / "model_et.joblib")      # Extra Trees
        models["GB"] = load(ARTIFACTS_DIR / "model_gb.joblib")      # Gradient Boosting
        models["XGB"] = load(ARTIFACTS_DIR / "model_xgb.joblib")    # XGBoost
        models["LGBM"] = load(ARTIFACTS_DIR / "model_lgbm.joblib")  # LightGBM
        
        # ============================================================
        # CARICAMENTO CATBOOST (formato proprietario .cbm)
        # ============================================================
        # CatBoost usa formato proprietario .cbm, non joblib
        # Necessita di inizializzare un CatBoostClassifier vuoto
        # e poi caricare i pesi tramite load_model()
        # ============================================================
        cat = CatBoostClassifier()
        cat.load_model(str(ARTIFACTS_DIR / "model_cat.cbm"))
        models["CAT"] = cat
        
        # ============================================================
        # CARICAMENTO MODELLO DEEP LEARNING (TensorFlow/Keras)
        # ============================================================
        # Il modello neural network usa TensorFlow/Keras formato .keras
        # Include architettura, pesi e configurazione ottimizzatore
        # ============================================================
        models["DL"] = tf.keras.models.load_model(ARTIFACTS_DIR / "model_dl.keras")
        
        # ============================================================
        # CARICAMENTO THRESHOLD OTTIMIZZATO
        # ============================================================
        # La soglia decisionale è stata ottimizzata dal Threshold Tournament
        # durante il training. Se non esiste, usa default 0.5
        # ============================================================
        threshold = 0.5  # Default se file non esiste
        
        thresh_path = ARTIFACTS_DIR / "threshold.txt"
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                threshold = float(f.read().strip())
        
        return preprocessor, models, threshold
    
    except Exception as e:
        # ============================================================
        # GESTIONE ERRORI
        # ============================================================
        # Errore bloccante: senza modelli o preprocessor non si può inferire
        # Il modulo termina con exit code 1
        # ============================================================
        console.print(f"[bold red]Load Error: {e}[/bold red]")
        console.print("[yellow]>>> Please ensure you have run Module 01 and 02 first.[/yellow]")
        sys.exit(1)


# ============================================================ #
# DATABASE HELPERS
# ============================================================ #
# Salva:
# - input clinico
# - predizione binaria
# - probabilità stimata
# ============================================================ #

def save_to_db(data, prediction, probability):
    """
    Salva i dati del paziente e la predizione nel database SQLite.
    
    Aggiunge automaticamente valori di default per colonne non raccolte
    durante l'input interattivo, e rimuove colonne di feature engineering
    che non sono parte dello schema database.
    
    Args:
        data: DataFrame con i dati del paziente (da get_interactive_input())
        prediction: Predizione binaria (0 o 1)
        probability: Probabilità stimata (0.0 - 1.0)
    """
    # Connessione context-manager (auto close)
    with get_db_connection() as conn:
        df_to_save = data.copy()
        
        # ============================================================
        # DEFINIZIONE COLONNE VALIDE NEL DATABASE
        # ============================================================
        # Lista completa delle colonne accettate dallo schema database
        # (escluse 'id' e 'Timestamp' che sono gestite automaticamente)
        # ============================================================
        valid_db_columns = [
            "State", "Sex", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays",
            "LastCheckupTime", "PhysicalActivities", "SleepHours", "RemovedTeeth",
            "HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer",
            "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis",
            "HadDiabetes", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
            "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
            "DifficultyErrands", "SmokerStatus", "ECigaretteUsage", "ChestScan",
            "RaceEthnicityCategory", "AgeCategory", "HeightInMeters", "WeightInKilograms",
            "BMI", "AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver",
            "TetanusLast10Tdap", "HighRiskLastYear", "CovidPos"
        ]
        
        # ============================================================
        # RIMOZIONE COLONNE NON VALIDE (FEATURE ENGINEERING)
        # ============================================================
        # Rimuove colonne di feature engineering che non sono nello schema
        # (es: GeneralHealth_Num, Sleep_Health_Ratio)
        # ============================================================
        columns_to_remove = [col for col in df_to_save.columns if col not in valid_db_columns]
        if columns_to_remove:
            df_to_save = df_to_save.drop(columns=columns_to_remove)
        
        # ============================================================
        # AGGIUNTA COLONNE MANCANTI (valori di default)
        # ============================================================
        # Lo schema database richiede molte colonne che non vengono
        # raccolte durante l'input interattivo. Aggiungiamo valori
        # di default per garantire compatibilità con lo schema.
        # ============================================================
        default_values = {
            "State": "",  # Stato non richiesto in input interattivo (stringa vuota per TEXT)
            "LastCheckupTime": "Within past year",  # Default comune
            "RemovedTeeth": "None of them",  # Default comune
            "HadHeartAttack": "No",  # Non richiesto (è il target predetto)
            "HadAngina": "No",  # Non richiesto (è parte del target)
            "HIVTesting": "No",  # Non richiesto in input
            "FluVaxLast12": "No",  # Non richiesto in input
            "PneumoVaxEver": "No",  # Non richiesto in input
            "TetanusLast10Tdap": "No",  # Non richiesto in input
            "HighRiskLastYear": "No",  # Non richiesto in input
            "CovidPos": "No"  # Non richiesto in input
        }
        
        # Aggiunge solo le colonne mancanti (non sovrascrive colonne esistenti)
        for col, default_val in default_values.items():
            if col not in df_to_save.columns:
                df_to_save[col] = default_val
        
        # Aggiunge colonne di output (non sono in valid_db_columns perché gestite separatamente)
        df_to_save['Prediction'] = int(prediction)
        df_to_save['Probability'] = float(probability)
        
        try:
            # Append nel database SQLite
            df_to_save.to_sql('patients', conn, if_exists='append', index=False)
            console.print("[dim][SUCCESS] Patient data saved to database[/dim]")
        except Exception as e:
            # Mostra errore invece di fallire silenziosamente
            # Questo aiuta a debuggare problemi di schema/compatibilità
            console.print(f"[yellow][WARNING] Could not save to database: {e}[/yellow]")
            console.print("[dim]Prediction completed, but data was not saved.[/dim]")


# ============================================================ #
# FEATURE ENGINEERING (IDENTICO AL TRAINING)
# ============================================================ #
# ATTENZIONE:
# ogni feature ingegnerizzata deve essere IDENTICA
# a quella usata nel Module 01
# ============================================================ #

def add_engineered_features(df):
    """
    Applica feature engineering identico a quello del training (Module 01).
    
    Questa funzione DEVE essere identica alla feature engineering usata durante
    il training per garantire coerenza tra training e inference.
    
    Feature create:
    1. GeneralHealth_Num: Convertisce GeneralHealth categoriale in numerico ordinale
       (Poor=1, Fair=2, Good=3, Very good=4, Excellent=5)
    2. Sleep_Health_Ratio: Rapporto SleepHours / (PhysicalHealthDays + 1)
       Rappresenta l'interazione tra qualità del sonno e salute fisica
    
    Args:
        df: DataFrame con dati paziente (deve contenere colonne necessarie)
    
    Returns:
        DataFrame: DataFrame con feature engineering aggiunte (modifica in-place del parametro)
    
    Note:
        - Le feature create qui vengono usate dal preprocessor ma poi rimosse
          prima del salvataggio nel database (non fanno parte dello schema DB)
        - Fillna(3) per GeneralHealth_Num significa che valori mancanti diventano "Good"
    """
    # ============================================================
    # 1. FEATURE: GeneralHealth_Num (categorico → ordinale numerico)
    # ============================================================
    # Mapping per convertire categorie ordinali in numeri
    # Permette ai modelli di interpretare l'ordine (Excellent > Poor)
    # ============================================================
    gen_health_map = {
        "Poor": 1,           # Peggiore salute
        "Fair": 2,           # Salute mediocre
        "Good": 3,           # Salute neutra/buona
        "Very good": 4,      # Salute molto buona
        "Excellent": 5       # Migliore salute
    }
    
    # Applica mapping solo se colonna esiste
    # Fillna(3) = "Good" come valore di default per missing
    if 'GeneralHealth' in df.columns:
        df['GeneralHealth_Num'] = df['GeneralHealth'].map(gen_health_map).fillna(3)

    # ============================================================
    # 2. FEATURE: Sleep_Health_Ratio (interazione sonno/salute)
    # ============================================================
    # Rapporto tra ore di sonno e giorni di cattiva salute fisica
    # Valori alti = buon sonno, poca malattia → buon segno
    # Valori bassi = poco sonno o molti giorni malati → fattore di rischio
    # (PhysicalHealthDays + 1) evita divisione per zero
    # ============================================================
    if 'SleepHours' in df.columns and 'PhysicalHealthDays' in df.columns:
        df['Sleep_Health_Ratio'] = df['SleepHours'] / (df['PhysicalHealthDays'] + 1)
        
    return df


# ============================================================ #
# INTERACTIVE CLI INPUT
# ============================================================ #
# Raccolta dati clinici conforme al dataset CDC 2022
# ============================================================ #

def get_interactive_input():
    """
    Raccolta dati clinici semplificata: input numerici e brevi invece di stringhe lunghe
    """
    console.print(Panel("[bold]Patient Clinical Data Entry[/bold]\n[dim]Press Enter for defaults[/dim]", style="cyan"))
    data = {}
    
    # -------------------------------
    # DEMOGRAFIA (semplificata)
    # -------------------------------
    console.print("\n[bold yellow]📋 DEMOGRAPHICS[/bold yellow]")
    
    # Sesso: 1=Male, 2=Female
    sex_choice = Prompt.ask("[green]Sex (1=Male, 2=Female)[/green]", choices=["1", "2"], default="1")
    data["Sex"] = "Male" if sex_choice == "1" else "Female"
    
    # Età: input numerico, calcola categoria automaticamente
    age = FloatPrompt.ask("[green]Age (years)[/green]", default=60.0)
    if age < 25:
        data["AgeCategory"] = "Age 18-24"
    elif age < 30:
        data["AgeCategory"] = "Age 25-29"
    elif age < 35:
        data["AgeCategory"] = "Age 30-34"
    elif age < 40:
        data["AgeCategory"] = "Age 35-39"
    elif age < 45:
        data["AgeCategory"] = "Age 40-44"
    elif age < 50:
        data["AgeCategory"] = "Age 45-49"
    elif age < 55:
        data["AgeCategory"] = "Age 50-54"
    elif age < 60:
        data["AgeCategory"] = "Age 55-59"
    elif age < 65:
        data["AgeCategory"] = "Age 60-64"
    elif age < 70:
        data["AgeCategory"] = "Age 65-69"
    elif age < 75:
        data["AgeCategory"] = "Age 70-74"
    elif age < 80:
        data["AgeCategory"] = "Age 75-79"
    else:
        data["AgeCategory"] = "Age 80 or older"
    
    # Razza/Etnia: numeri invece di stringhe lunghe
    race_map = {
        "1": "White only, Non-Hispanic",
        "2": "Black only, Non-Hispanic",
        "3": "Hispanic",
        "4": "Asian only, Non-Hispanic",
        "5": "Multiracial, Non-Hispanic",
        "6": "Other"
    }
    race_choice = Prompt.ask("[green]Race/Ethnicity (1=White, 2=Black, 3=Hispanic, 4=Asian, 5=Multi, 6=Other)[/green]", 
                              choices=["1", "2", "3", "4", "5", "6"], default="1")
    data["RaceEthnicityCategory"] = race_map[race_choice]
    
    # Altezza e peso (numerici, già ok)
    data["HeightInMeters"] = FloatPrompt.ask("[green]Height (meters, e.g. 1.75)[/green]", default=1.75)
    data["WeightInKilograms"] = FloatPrompt.ask("[green]Weight (kg, e.g. 80)[/green]", default=80.0)
    data["BMI"] = data["WeightInKilograms"] / (data["HeightInMeters"] ** 2)  # Calcolo BMI automatico
    
    # -------------------------------
    # STATO DI SALUTE (semplificato)
    # -------------------------------
    console.print("\n[bold yellow]🏥 HEALTH STATUS[/bold yellow]")
    
    # Salute generale: numeri invece di parole
    health_map = {"1": "Poor", "2": "Fair", "3": "Good", "4": "Very good", "5": "Excellent"}
    health_choice = Prompt.ask("[green]General Health (1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent)[/green]", 
                                choices=["1", "2", "3", "4", "5"], default="3")
    data["GeneralHealth"] = health_map[health_choice]
    
    # Giorni di cattiva salute (numerici, già ok)
    data["PhysicalHealthDays"] = FloatPrompt.ask("[green]Days of poor physical health (last 30 days)[/green]", default=0.0)
    data["MentalHealthDays"] = FloatPrompt.ask("[green]Days of poor mental health (last 30 days)[/green]", default=0.0)
    data["SleepHours"] = FloatPrompt.ask("[green]Average sleep hours per night[/green]", default=7.0)
    
    # -------------------------------
    # ABITUDINI (semplificate)
    # -------------------------------
    console.print("\n[bold yellow]🚭 LIFESTYLE[/bold yellow]")
    
    # Attività fisica: Y/N
    data["PhysicalActivities"] = Prompt.ask("[green]Physical activities in last 30 days? (Y/N)[/green]", 
                                             choices=["Y", "N", "y", "n", "Yes", "No"], default="Y")
    if data["PhysicalActivities"].upper() in ["Y", "YES"]:
        data["PhysicalActivities"] = "Yes"
    else:
        data["PhysicalActivities"] = "No"
    
    # Fumo: numeri invece di frasi lunghe
    smoke_map = {
        "1": "Never smoked",
        "2": "Former smoker",
        "3": "Current smoker - now smokes some days",
        "4": "Current smoker - now smokes every day"
    }
    smoke_choice = Prompt.ask("[green]Smoking (1=Never, 2=Former, 3=Some days, 4=Every day)[/green]", 
                               choices=["1", "2", "3", "4"], default="1")
    data["SmokerStatus"] = smoke_map[smoke_choice]
    
    # E-Cigarette: numeri
    ecig_map = {
        "1": "Never used e-cigarettes in my entire life",
        "2": "Not at all (right now)",
        "3": "Use them some days",
        "4": "Use them every day"
    }
    ecig_choice = Prompt.ask("[green]E-Cigarette (1=Never, 2=Not now, 3=Some days, 4=Every day)[/green]", 
                              choices=["1", "2", "3", "4"], default="1")
    data["ECigaretteUsage"] = ecig_map[ecig_choice]
    
    # Alcol: Y/N
    alcohol = Prompt.ask("[green]Alcohol drinker? (Y/N)[/green]", choices=["Y", "N", "y", "n"], default="N")
    data["AlcoholDrinkers"] = "Yes" if alcohol.upper() == "Y" else "No"

    # -------------------------------
    # CONDIZIONI MEDICHE (semplificate: default No, chiedi solo se presenti)
    # -------------------------------
    console.print("\n[bold yellow]💊 MEDICAL CONDITIONS[/bold yellow]")
    console.print("[dim]Press Enter for 'No' (no condition)[/dim]")
    
    # Inizializza tutte a "No"
    conditions_map = {
        "Stroke": "HadStroke",
        "Asthma": "HadAsthma",
        "Skin Cancer": "HadSkinCancer",
        "COPD (Chronic Obstructive Pulmonary Disease)": "HadCOPD",
        "Depression": "HadDepressiveDisorder",
        "Kidney Disease": "HadKidneyDisease",
        "Arthritis": "HadArthritis",
        "Diabetes": "HadDiabetes",
        "Deaf/Hard of Hearing": "DeafOrHardOfHearing",
        "Blind/Vision Difficulty": "BlindOrVisionDifficulty",
        "Difficulty Concentrating": "DifficultyConcentrating",
        "Difficulty Walking": "DifficultyWalking",
        "Difficulty Dressing/Bathing": "DifficultyDressingBathing",
        "Difficulty Errands": "DifficultyErrands",
        "Chest Scan (CT/CAT)": "ChestScan"
    }
    
    # Chiedi solo se presenti (Y/N, default No)
    for display_name, field_name in conditions_map.items():
        answer = Prompt.ask(f"[green]{display_name}?[/green]", choices=["Y", "N", "y", "n", ""], default="N")
        data[field_name] = "Yes" if answer.upper() == "Y" else "No"

    # Ritorna DataFrame a singola riga
    return pd.DataFrame([data])


# ============================================================ #
# ENSEMBLE PREDICTION
# ============================================================ #

def predict_ensemble(preprocessor, models, threshold):
    """
    Esegue predizione per un singolo paziente tramite ensemble di modelli.
    
    Processo completo:
    1. Raccolta dati clinici tramite CLI interattiva
    2. Feature engineering (coerente con training)
    3. Preprocessing (scaling + encoding)
    4. Predizioni da 8 modelli ML/DL
    5. Combinazione pesata delle probabilità
    6. Decisione binaria basata su threshold ottimizzata
    7. Visualizzazione risultato e salvataggio nel database
    
    Args:
        preprocessor: ColumnTransformer fittato (preprocessing pipeline)
        models: Dizionario con modelli addestrati ["LR", "RF", "ET", "GB", "XGB", "LGBM", "CAT", "DL"]
        threshold: Soglia decisionale ottimizzata dal Threshold Tournament (float 0-1)
    
    Note:
        - Usa pesi esperti basati su performance attesa dei modelli
        - LR ha peso 0 (non usato nell'ensemble finale)
        - XGB, LGBM, CAT hanno peso 4 (best-in-class per tabular data)
        - DL, RF, GB, ET hanno peso 2-3 (complementari)
    """
    # ============================================================
    # STEP 1: RACCOLTA DATI CLINICI
    # ============================================================
    # Input interattivo tramite CLI con Rich prompts
    # Ritorna DataFrame con una singola riga (paziente)
    # ============================================================
    raw_df = get_interactive_input()
    
    # ============================================================
    # STEP 2: FEATURE ENGINEERING
    # ============================================================
    # Applica feature engineering identico al training:
    # - GeneralHealth_Num (categorico → ordinale)
    # - Sleep_Health_Ratio (interazione sonno/salute)
    # ============================================================
    df = add_engineered_features(raw_df)
    
    try:
        # ============================================================
        # STEP 3: PREPROCESSING
        # ============================================================
        # Trasforma dati raw in formato pronto per i modelli:
        # - StandardScaler per feature numeriche (media 0, std 1)
        # - OneHotEncoder per feature categoriche (encoding binario)
        # ============================================================
        X = preprocessor.transform(df)
        
        # ============================================================
        # STEP 4: PREDIZIONI DA OGNI MODELLO
        # ============================================================
        # Ogni modello produce probabilità classe positiva (malattia)
        # Modelli ML: predict_proba(X)[0][1] = probabilità classe 1
        # Modello DL: predict(X)[0][0] = output sigmoid (già probabilità)
        # ============================================================
        p_lr = models["LR"].predict_proba(X)[0][1]    # Logistic Regression
        p_rf = models["RF"].predict_proba(X)[0][1]    # Random Forest
        p_et = models["ET"].predict_proba(X)[0][1]    # Extra Trees
        p_gb = models["GB"].predict_proba(X)[0][1]    # Gradient Boosting
        p_xgb = models["XGB"].predict_proba(X)[0][1]  # XGBoost
        p_lgbm = models["LGBM"].predict_proba(X)[0][1] # LightGBM
        p_cat = models["CAT"].predict_proba(X)[0][1]  # CatBoost
        p_dl = models["DL"].predict(X, verbose=0)[0][0] # Deep Learning (output già probabilità)
        
        # ============================================================
        # STEP 5: ENSEMBLE PESATO (MEDIA PONDERATA)
        # ============================================================
        # Combinazione delle probabilità con pesi esperti
        # I pesi sono basati su performance attesa:
        # - Boosting (XGB, LGBM, CAT): peso 4 (best-in-class)
        # - Extra Trees: peso 3 (buono per pattern non-lineari)
        # - Random Forest, GB, DL: peso 2 (complementari)
        # - Logistic Regression: peso 0 (non usato)
        # ============================================================
        weights = {
            "LR": 0,   # Non usato nell'ensemble (peso 0)
            "RF": 2,   # Random Forest (complementare)
            "ET": 3,   # Extra Trees (buono per pattern non-lineari)
            "GB": 2,   # Gradient Boosting (complementare)
            "XGB": 4,  # XGBoost (best-in-class per tabular)
            "LGBM": 4, # LightGBM (best-in-class per tabular)
            "CAT": 4,  # CatBoost (best-in-class per tabular)
            "DL": 2    # Deep Learning (complementare)
        }
        
        total_w = sum(weights.values())  # Peso totale per normalizzazione
        
        # Media pesata delle probabilità
        avg_prob = (
            p_lr*0 + p_rf*2 + p_et*3 + p_gb*2 + 
            p_xgb*4 + p_lgbm*4 + p_cat*4 + p_dl*2
        ) / total_w

        # ============================================================
        # STEP 6: DECISIONE BINARIA
        # ============================================================
        # Applica threshold ottimizzata dal Threshold Tournament
        # Se probabilità >= threshold → High Risk (pred=1)
        # Se probabilità < threshold → Low Risk (pred=0)
        # ============================================================
        pred = 1 if avg_prob >= threshold else 0
        
        # Stile per output colorato (rosso per High Risk, verde per Low Risk)
        style = "bold red" if pred == 1 else "bold green"
        msg = "High Risk (Detection)" if pred == 1 else "Low Risk (Healthy)"
        
        # ============================================================
        # STEP 7: VISUALIZZAZIONE RISULTATO
        # ============================================================
        # Pannello Rich con risultato, confidence, threshold usata
        # ============================================================
        console.print(Panel(
            f"[bold]Mega-Ensemble Prediction:[/bold] [{style}]{msg}[/]\n"
            f"[bold]Confidence:[/bold] {avg_prob:.2%}\n"
            f"[dim]Threshold (Tournament Best): {threshold:.2%}[/dim]\n"
            f"[dim]Models: XGB, LGBM, CAT, ET, RF, GB, DL[/dim]",
            title="Optimized Result",
            border_style=style
        ))
        
        # ============================================================
        # STEP 8: SALVATAGGIO NEL DATABASE
        # ============================================================
        # Salva dati paziente, predizione e probabilità in SQLite
        # Usa raw_df (senza feature engineering) per compatibilità schema DB
        # ============================================================
        save_to_db(raw_df, pred, avg_prob)

    except Exception as e:
        # Gestione errori durante predizione (es: feature mancanti, modelli corrotti)
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ============================================================ #
# BATCH PREDICTION (TUTTI I PAZIENTI NEL DATABASE)
# ============================================================ #
# Predice per tutti i pazienti nel database che non hanno
# ancora una predizione, o aggiorna tutte le predizioni
# ============================================================ #

def predict_batch(preprocessor, models, threshold):
    """
    Esegue predizioni batch su tutti i pazienti nel database.
    
    Args:
        preprocessor: Preprocessor fittato
        models: Dizionario di modelli addestrati
        threshold: Soglia decisionale ottimizzata
    """
    console.print("[bold cyan]Loading patients from database...[/bold cyan]")
    
    try:
        # ============================================================
        # STEP 1: CARICAMENTO PAZIENTI DAL DATABASE
        # ============================================================
        # Legge tutti i pazienti presenti nel database SQLite
        # Include anche colonne di output (Prediction, Probability) se esistono
        # ============================================================
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM patients", conn)
            
        if len(df) == 0:
            console.print("[yellow]No patients found in database. Use option 4 to add patients first.[/yellow]")
            return
        
        console.print(f"[green]Found {len(df)} patients in database[/green]")
        
        # ============================================================
        # STEP 2: PREPARAZIONE FEATURE (rimozione colonne output)
        # ============================================================
        # Rimuove colonne di output/metadati per permettere ri-predizione
        # e feature engineering pulito. Le colonne di output vengono
        # aggiornate dopo la predizione.
        # ============================================================
        output_cols = ['Prediction', 'Probability', 'Timestamp', 'id']
        feature_cols = [c for c in df.columns if c not in output_cols]
        df_features = df[feature_cols].copy()
        
        # ============================================================
        # STEP 3: FEATURE ENGINEERING
        # ============================================================
        # Applica feature engineering identico al training:
        # - GeneralHealth_Num (categorico → ordinale)
        # - Sleep_Health_Ratio (interazione sonno/salute)
        # ============================================================
        df_features = add_engineered_features(df_features)
        
        # ============================================================
        # STEP 4: PREPROCESSING
        # ============================================================
        # Trasforma dati raw in formato pronto per i modelli
        # (StandardScaler + OneHotEncoder)
        # ============================================================
        X = preprocessor.transform(df_features)
        
        # ============================================================
        # STEP 5: PREDIZIONI ENSEMBLE PER BATCH
        # ============================================================
        # Calcola probabilità da ogni modello per tutti i pazienti
        # simultaneamente (batch prediction efficiente)
        # ============================================================
        console.print("[cyan]Computing predictions for all patients...[/cyan]")
        
        # Probabilità per ogni modello (array con una probabilità per paziente)
        probs = {}
        for name, model in models.items():
            if name == "DL":
                # Deep Learning: predict() ritorna già probabilità (sigmoid output)
                probs[name] = model.predict(X, verbose=0).ravel()
            else:
                # Modelli ML: predict_proba() ritorna [prob_0, prob_1]
                # Prendiamo prob_1 (classe positiva = malattia)
                probs[name] = model.predict_proba(X)[:, 1]
        
        # ============================================================
        # STEP 6: ENSEMBLE PESATO (identico a predict_ensemble)
        # ============================================================
        # Combinazione pesata delle probabilità con stessi pesi del training
        # ============================================================
        weights = {
            "LR": 0, "RF": 2, "ET": 3, "GB": 2, 
            "XGB": 4, "LGBM": 4, "CAT": 4, "DL": 2
        }
        
        total_w = sum(weights.values())
        ensemble_proba = np.zeros(len(X), dtype=float)
        
        # Somma pesata delle probabilità
        for name, w in weights.items():
            if w > 0:
                ensemble_proba += probs[name] * w
        
        # Normalizzazione per somma pesi
        ensemble_proba /= total_w
        
        # ============================================================
        # STEP 7: DECISIONI BINARIE
        # ============================================================
        # Applica threshold per convertire probabilità in predizioni binarie
        # ============================================================
        predictions = (ensemble_proba >= threshold).astype(int)
        
        # ============================================================
        # STEP 8: AGGIORNAMENTO DATABASE
        # ============================================================
        # Aggiorna ogni record con nuova predizione e probabilità
        # Timestamp viene aggiornato automaticamente a CURRENT_TIMESTAMP
        # ============================================================
        console.print("[cyan]Updating database with predictions...[/cyan]")
        
        with get_db_connection() as conn:
            # Aggiorna ogni record con predizione e probabilità
            # UPDATE tramite WHERE id garantisce update corretto anche con ID non sequenziali
            for idx, (pred, prob) in enumerate(zip(predictions, ensemble_proba)):
                conn.execute("""
                    UPDATE patients 
                    SET Prediction = ?, Probability = ?, Timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (int(pred), float(prob), int(df.iloc[idx]['id'])))
            
            conn.commit()  # Commit finale per salvare tutte le modifiche
        
        # ============================================================
        # STEP 9: STATISTICHE E OUTPUT
        # ============================================================
        # Calcola statistiche finali e mostra risultati batch
        # ============================================================
        num_positive = sum(predictions)  # Numero High Risk (pred=1)
        num_negative = len(predictions) - num_positive  # Numero Low Risk (pred=0)
        
        console.print(Panel(
            f"[bold]Batch Prediction Complete[/bold]\n\n"
            f"[green]Total Patients:[/green] {len(df)}\n"
            f"[red]High Risk (Disease):[/red] {num_positive} ({num_positive/len(df):.1%})\n"
            f"[green]Low Risk (Healthy):[/green] {num_negative} ({num_negative/len(df):.1%})\n\n"
            f"[dim]Threshold used: {threshold:.3f}[/dim]",
            title="Batch Results",
            border_style="cyan"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Batch prediction error: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ============================================================ #
# MAIN ENTRY POINT
# ============================================================ #

if __name__ == "__main__":
    init_db() # Inizializza database se non esiste
    preprocessor, models, threshold = load_resources() # Carica risorse
    
    # Gestione flag da command line
    if "--batch" in sys.argv:
        # Modalità batch: predice per tutti i pazienti nel database
        predict_batch(preprocessor, models, threshold)
    elif "--history" in sys.argv:
        # TODO: Implementare visualizzazione history
        console.print("[yellow]History feature not yet implemented[/yellow]")
    else:
        # Modalità interattiva: predizione per un singolo paziente
        predict_ensemble(preprocessor, models, threshold)
