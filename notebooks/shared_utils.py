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
