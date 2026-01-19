# ============================================================ #
# SHARED UTILITIES - Heart Disease Prediction                  #
# ============================================================ #

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.theme import Theme
import pandas as pd
import sqlite3
import numpy as np  # Necessario per EnsembleWrapper

# --- 1. RICH CONSOLE SETUP ---
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "header": "bold magenta",
    "input": "bold blue"
})
console = Console(theme=custom_theme)

# --- 2. ENVIRONMENT SETUP ---
def setup_environment():
    """Sets up critical environment variables for macOS/Anaconda stability."""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_METAL_DEVICE_NS'] = '0' # Disable Metal usage for TF plugin stability

# --- 3. PATH DEFINITIONS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DB_PATH = PROJECT_ROOT / "patients_data.db"
DATASET_PATH = DATA_DIR / "heart_2022_no_nans.csv"

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 4. DATA MAPPINGS (NEW 2022 SCHEMA) ---
CLINICAL_GUIDE = {
    "Sex": "Biological Sex (Male/Female).",
    "GeneralHealth": "Self-reported health (Excellent to Poor).",
    "PhysicalHealthDays": "Days of poor physical health in past 30 days.",
    "MentalHealthDays": "Days of poor mental health in past 30 days.",
    "LastCheckupTime": "Time since last checkup.",
    "PhysicalActivities": "Exercise in past 30 days (Yes/No).",
    "SleepHours": "Average hours of sleep.",
    "HadAsthma": "History of Asthma (Yes/No).",
    "HadSkinCancer": "History of Skin Cancer (Yes/No).",
    "HadCOPD": "History of COPD (Yes/No).",
    "HadDepressiveDisorder": "History of Depressive Disorder (Yes/No).",
    "HadKidneyDisease": "History of Kidney Disease (Yes/No).",
    "HadArthritis": "History of Arthritis (Yes/No).",
    "HadDiabetes": "History of Diabetes (Yes/No/Borderline).",
    "SmokerStatus": "Smoking history.",
    "ECigaretteUsage": "E-Cigarette usage.",
    "ChestScan": "Had CT/CAT scan of chest (Yes/No).",
    "AgeCategory": "Age Range.",
    "BMI": "Body Mass Index.",
    "AlcoholDrinkers": "Alcohol consumption (Yes/No).",
    "HIVTesting": "Ever tested for HIV (Yes/No).",
    "CovidPos": "Tested positive for COVID-19 (Yes/No)."
}

# --- 5. DATABASE UTILITIES ---
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Updated Schema for 2022 Dataset
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                State TEXT, Sex TEXT, GeneralHealth TEXT, PhysicalHealthDays REAL, 
                MentalHealthDays REAL, LastCheckupTime TEXT, PhysicalActivities TEXT, 
                SleepHours REAL, RemovedTeeth TEXT, HadHeartAttack TEXT, HadAngina TEXT, 
                HadStroke TEXT, HadAsthma TEXT, HadSkinCancer TEXT, HadCOPD TEXT, 
                HadDepressiveDisorder TEXT, HadKidneyDisease TEXT, HadArthritis TEXT, 
                HadDiabetes TEXT, DeafOrHardOfHearing TEXT, BlindOrVisionDifficulty TEXT, 
                DifficultyConcentrating TEXT, DifficultyWalking TEXT, 
                DifficultyDressingBathing TEXT, DifficultyErrands TEXT, SmokerStatus TEXT, 
                ECigaretteUsage TEXT, ChestScan TEXT, RaceEthnicityCategory TEXT, 
                AgeCategory TEXT, HeightInMeters REAL, WeightInKilograms REAL, 
                BMI REAL, AlcoholDrinkers TEXT, HIVTesting TEXT, FluVaxLast12 TEXT, 
                PneumoVaxEver TEXT, TetanusLast10Tdap TEXT, HighRiskLastYear TEXT, CovidPos TEXT,
                Prediction INTEGER, Probability REAL, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP 
            )
        ''')
        conn.commit()

# --- 6. COMMON UI FUNCTIONS ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str):
    console.print(f"\n[bold magenta]{'=' * 60}[/bold magenta]")
    console.print(f"[bold white]{title.center(60)}[/bold white]")
    console.print(f"[bold magenta]{'=' * 60}[/bold magenta]\n")

# --- 7. MODEL CHECKING UTILITY ---
def check_models_exist():
    """
    Verifica se i modelli addestrati esistono già nella cartella artifacts/.
    
    Questa funzione controlla la presenza di tutti i file necessari per considerare
    il training completo e i modelli pronti per inference e explainability.
    
    File verificati:
    - preprocessor.joblib: Pipeline preprocessing (StandardScaler + OneHotEncoder)
    - model_*.joblib: Modelli ML classici (LR, RF, ET, GB, XGB, LGBM)
    - model_cat.cbm: Modello CatBoost (formato proprietario)
    - model_dl.keras: Modello Deep Learning (TensorFlow/Keras)
    - best_model_unified.joblib: Ensemble wrapper unificato per SHAP
    - model_type.txt: Tipo modello attivo ("ensemble" o "keras")
    - threshold.txt: Soglia decisionale ottimizzata dal Threshold Tournament
    
    Returns:
        bool: True se tutti i file necessari esistono, False altrimenti
    
    Note:
        - Se anche un solo file manca, la funzione ritorna False
        - Non verifica la validità/correttezza dei file, solo la presenza
        - Usata in main.py per decidere se offrire opzione di skip training
    """
    # Lista completa dei file richiesti per un training completo
    # Ogni file è essenziale per il funzionamento di inference e explainability
    required_files = [
        "preprocessor.joblib",           # Pipeline preprocessing (necessario per trasformare nuovi dati)
        "model_lr.joblib",               # Logistic Regression
        "model_rf.joblib",               # Random Forest
        "model_et.joblib",               # Extra Trees
        "model_gb.joblib",               # Gradient Boosting
        "model_xgb.joblib",              # XGBoost
        "model_lgbm.joblib",             # LightGBM
        "model_cat.cbm",                 # CatBoost (formato .cbm, non .joblib)
        "model_dl.keras",                # Neural Network (TensorFlow/Keras)
        "best_model_unified.joblib",     # Ensemble wrapper unificato (per SHAP e inference semplificata)
        "model_type.txt",                # Tipo modello ("ensemble" o "keras")
        "threshold.txt"                  # Soglia ottimizzata dal Threshold Tournament
    ]
    
    # Verifica esistenza di ogni file richiesto
    missing_files = []
    for filename in required_files:
        filepath = ARTIFACTS_DIR / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    # Ritorna True solo se tutti i file esistono (nessun file mancante)
    return len(missing_files) == 0

# --- 8. ENSEMBLE WRAPPER CLASS ---
# Classe condivisa per wrapper ensemble, necessaria per deserializzazione joblib
# Deve essere in shared_utils per essere importabile da tutti i moduli

class EnsembleWrapper:
    """
    Wrapper per l'ensemble di modelli che permette di trattare
    l'intero ensemble come un singolo modello per SHAP e explainability.
    
    Questa classe DEVE essere definita in shared_utils perché joblib
    la cerca nel modulo di origine quando deserializza best_model_unified.joblib
    """
    def __init__(self, models, weights):
        """
        Inizializza l'ensemble con modelli e pesi.
        
        Args:
            models: Dizionario di modelli addestrati
            weights: Dizionario di pesi per ogni modello
        """
        self.models = models  # Dizionario modelli (LR, RF, ET, GB, XGB, LGBM, CAT, DL)
        self.weights = weights  # Dizionario pesi per ogni modello
        self.total_weight = sum(weights.values())  # Peso totale per normalizzazione
    
    def predict(self, X):
        """
        Predice le probabilità usando l'ensemble pesato.
        
        Args:
            X: Feature matrix preprocessata (array numpy o array-like)
            
        Returns:
            Array di probabilità per classe positiva (shape: [n_samples, 1])
        """
        ensemble_proba = np.zeros(X.shape[0], dtype=float)  # Inizializza array probabilità
        
        # Calcola probabilità pesata per ogni modello
        for name, model in self.models.items():
            if self.weights[name] > 0:  # Considera solo modelli con peso > 0
                if name == "DL":  # Deep Learning usa API diversa
                    proba = model.predict(X, verbose=0).ravel()  # Predizione diretta (già probabilità)
                else:  # Modelli scikit-learn / boosting
                    proba = model.predict_proba(X)[:, 1]  # Probabilità classe positiva
                ensemble_proba += proba * self.weights[name]  # Somma pesata
        
        ensemble_proba /= self.total_weight  # Normalizza per somma pesi
        return ensemble_proba.reshape(-1, 1)  # Ritorna shape [n_samples, 1]
    
    def predict_proba(self, X):
        """
        Ritorna probabilità per entrambe le classi (scikit-learn API).
        
        Args:
            X: Feature matrix preprocessata
            
        Returns:
            Array [prob_classe_0, prob_classe_1] (shape: [n_samples, 2])
        """
        proba_pos = self.predict(X).ravel()  # Probabilità classe positiva (malattia)
        proba_neg = 1 - proba_pos  # Probabilità classe negativa (sano) = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])  # Stack: [prob_0, prob_1]
