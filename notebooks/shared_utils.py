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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Optional: force CPU if needed widely
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 3. PATH DEFINITIONS ---
# Assuming this file is in <ROOT>/notebooks/
# PROJECT_ROOT is the parent of the directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DB_PATH = PROJECT_ROOT / "patients_data.db"
DATASET_PATH = DATA_DIR / "heart_disease.csv"

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 4. DATA MAPPINGS ---
CLINICAL_GUIDE = {
    "Age": "Patient's age in years.",
    "Gender": "Biological gender (Male/Female).",
    "Blood Pressure": "Systolic blood pressure (mm Hg).",
    "Cholesterol Level": "Total serum cholesterol level (mg/dl).",
    "Exercise Habits": "Regular physical activity level (Low, Medium, High).",
    "Smoking": "Current smoking status (Yes/No).",
    "Family Heart Disease": "History of heart disease in family (Yes/No).",
    "Diabetes": "Whether the patient has a diabetes diagnosis (Yes/No).",
    "BMI": "Body Mass Index (weight / height^2).",
    "High Blood Pressure": "Pre-existing hypertension (Yes/No).",
    "Low HDL Cholesterol": "Presence of low 'good' cholesterol (Yes/No).",
    "High LDL Cholesterol": "Presence of high 'bad' cholesterol (Yes/No).",
    "Alcohol Consumption": "Alcohol intake level (None, Low, Medium, High).",
    "Stress Level": "Reported psychological stress (Low, Medium, High).",
    "Sleep Hours": "Average hours of sleep per night.",
    "Sugar Consumption": "Dietary sugar intake (Low, Medium, High).",
    "Triglyceride Level": "Serum triglyceride level (mg/dl).",
    "Fasting Blood Sugar": "Blood sugar level after fasting.",
    "CRP Level": "C-reactive protein level (inflammation marker).",
    "Homocysteine Level": "Homocysteine level (vascular health marker)."
}

MAPPINGS = {
    "Binary": {"Yes": 1, "No": 0, "Male": 1, "Female": 0},
    "Ordinal_Basic": {"Low": 0, "Medium": 1, "High": 2, "None": 0}, 
    "Ordinal_Alcohol": {"None": 0, "Low": 1, "Medium": 2, "High": 3}
}

# --- 5. DATABASE UTILITIES ---
def get_db_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database schema if it doesn't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                Age REAL, Gender TEXT, [Blood Pressure] REAL, [Cholesterol Level] REAL, 
                [Exercise Habits] TEXT, Smoking TEXT, [Family Heart Disease] TEXT, 
                Diabetes TEXT, BMI REAL, [High Blood Pressure] TEXT, 
                [Low HDL Cholesterol] TEXT, [High LDL Cholesterol] TEXT, 
                [Alcohol Consumption] TEXT, [Stress Level] TEXT, [Sleep Hours] REAL, 
                [Sugar Consumption] TEXT, [Triglyceride Level] REAL, 
                [Fasting Blood Sugar] REAL, [CRP Level] REAL, [Homocysteine Level] REAL,
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
