# ============================================================ # Module 04 – Inference, Database & Clinical Support
# NOTEBOOK 4 – INTERACTIVE PREDICTION & SQL STORAGE           # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
# --- CRITICAL STABILITY & macOS FIXES (MUST BE BEFORE ANY OTHER IMPORTS) ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1' # Limit MKL threads
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU
# ---------------------------------------------------------------------------

print("Importing system libraries...") # Diagnostic print
import sys # Import sys to check the Python environment
import sqlite3 # Import sqlite3 for database management
from pathlib import Path # Import Path for robust filesystem path manipulation

print("Importing data science libraries...") # Diagnostic print
import pandas as pd # Import pandas for data handling
import numpy as np # Import numpy for numerical operations

print("Importing Machine Learning tools...") # Diagnostic print
from joblib import load # Import load from joblib to retrieve saved models
import tensorflow as tf # Import tensorflow for Keras model loading

print("All libraries imported successfully.") # Diagnostic print

# Define project paths                                                  # Path Definition Section
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Identify the project root directory
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path
DB_PATH = PROJECT_ROOT / "patients_data.db" # Define the path for the SQL database

# Clinical Glossary and Mapping Logic                                   # Mapping Section
CLINICAL_GUIDE = { # Define glossary
    "Age": "Patient's age in years.",
    "Gender": "Male or Female.",
    "Blood Pressure": "Systolic blood pressure (mm Hg).",
    "Cholesterol Level": "Total cholesterol (mg/dl).",
    "Exercise Habits": "Physical activity level (Low, Medium, High).",
    "Smoking": "Current smoker (Yes or No).",
    "Family Heart Disease": "Family history of heart disease (Yes or No).",
    "Diabetes": "Diabetes diagnosis (Yes or No).",
    "BMI": "Body Mass Index.",
    "High Blood Pressure": "History of high BP (Yes or No).",
    "Low HDL Cholesterol": "Low 'good' cholesterol (Yes or No).",
    "High LDL Cholesterol": "High 'bad' cholesterol (Yes or No).",
    "Alcohol Consumption": "Alcohol intake level (None, Low, Medium, High).",
    "Stress Level": "Reported stress level (Low, Medium, High).",
    "Sleep Hours": "Average nightly sleep hours.",
    "Sugar Consumption": "Dietary sugar level (Low, Medium, High).",
    "Triglyceride Level": "Serum triglycerides (mg/dl).",
    "Fasting Blood Sugar": "Blood sugar after fasting.",
    "CRP Level": "C-reactive protein level.",
    "Homocysteine Level": "Homocysteine level."
} # End glossary

MAPPINGS = { # Define conversion dictionaries
    "Binary": {"Yes": 1, "No": 0, "Male": 1, "Female": 0},
    "Ordinal_Basic": {"Low": 0, "Medium": 1, "High": 2},
    "Ordinal_Alcohol": {"None": 0, "Low": 1, "Medium": 2, "High": 3}
} # End mappings

def init_db(): # Define function to initialize the SQL database
    conn = sqlite3.connect(DB_PATH) # Connect
    cursor = conn.cursor() # Create cursor
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
    ''') # Create table with new schema
    conn.commit() # Commit
    conn.close() # Close

def save_to_db(data, prediction, probability): # Define function to save data to SQL
    conn = sqlite3.connect(DB_PATH) # Connect
    df_to_save = data.copy() # Copy
    df_to_save['Prediction'] = int(prediction) # Add pred
    df_to_save['Probability'] = float(probability) # Add prob
    df_to_save.to_sql('patients', conn, if_exists='append', index=False) # Append
    conn.close() # Close

def get_interactive_input(): # Define function for manual patient data entry
    print("\n--- NEW PATIENT DATA ENTRY ---") # Print header
    raw_data = {} # Raw input for DB
    numeric_data = {} # Converted for Model
    for key, explanation in CLINICAL_GUIDE.items(): # Iterate glossary
        print(f"\n[GUIDE] {key}: {explanation}") # Print guide
        while True: # Validation
            val = input(f"Enter {key}: ").strip() # Get input
            if not val: continue # Require input
            raw_data[key] = [val] # Store raw
            try: # Try conversion
                if key in ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]:
                    numeric_data[key] = [float(val)] # Num
                elif key in ["Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol", "Gender"]:
                    numeric_data[key] = [MAPPINGS["Binary"][val]] # Bin
                elif key == "Alcohol Consumption":
                    numeric_data[key] = [MAPPINGS["Ordinal_Alcohol"][val]] # Alc
                else: numeric_data[key] = [MAPPINGS["Ordinal_Basic"][val]] # Basic
                break # Success
            except KeyError: print("Invalid category name.") # Cat error
            except ValueError: print("Enter a numeric value.") # Num error
    return pd.DataFrame(raw_data), pd.DataFrame(numeric_data) # Return DataFrames

def predict_single(preprocessor, model, model_type): # Define single prediction flow
    raw_df, numeric_df = get_interactive_input() # Get input
    try: # Inference block
        processed = preprocessor.transform(numeric_df) # Preprocess
        if model_type == "keras": # If Deep Learning
            prob = model.predict(processed, verbose=0)[0][0] # Get prob
            pred = 1 if prob >= 0.5 else 0 # Threshold
        else: # If Classical ML
            pred = model.predict(processed)[0] # Get class
            prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else 0.0 # Prob

        print("\n--- PREDICTION RESULTS ---") # Print
        print(f"RESULT: {'Presence Detected' if pred == 1 else 'No Disease detected'}") # Out
        print(f"Model Probability: {prob:.2%}") # Out
        save_to_db(raw_df, pred, prob) # Persist
        print("\n[DB] Record saved.") # Conf
    except Exception as e: print(f"Prediction Error: {e}") # Err

def batch_predict_from_db(preprocessor, model, model_type): # Define batch prediction
    conn = sqlite3.connect(DB_PATH) # Connect
    try: # Block
        df = pd.read_sql_query("SELECT * FROM patients", conn) # Load
        if df.empty: return print("DB Empty.") # Check
        df_numeric = df.copy().drop(columns=['id', 'Prediction', 'Probability', 'Timestamp']) # Clean
        for col in df_numeric.columns: # Re-map strings to numbers
            if col in ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]:
                df_numeric[col] = df_numeric[col].astype(float)
            elif col in ["Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol", "Gender"]:
                df_numeric[col] = df_numeric[col].map(MAPPINGS["Binary"])
            elif col == "Alcohol Consumption":
                df_numeric[col] = df_numeric[col].map(MAPPINGS["Ordinal_Alcohol"])
            else: df_numeric[col] = df_numeric[col].map(MAPPINGS["Ordinal_Basic"])

        processed = preprocessor.transform(df_numeric) # Preprocess
        if model_type == "keras": # DL
            df['New_Prob'] = model.predict(processed, verbose=0).ravel()
            df['New_Pred'] = (df['New_Prob'] >= 0.5).astype(int)
        else: # Sklearn
            df['New_Pred'] = model.predict(processed)
        print("\n--- BATCH RESULTS ---") # Header
        print(df[['id', 'Age', 'Gender', 'Prediction', 'New_Pred']]) # Show
    except Exception as e: print(f"Batch Error: {e}") # Err
    finally: conn.close() # Close

def main(): # Local main
    init_db() # Ensure DB
    try: # Loading block
        preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib") # Load prep
        with open(ARTIFACTS_DIR / "model_type.txt", "r") as f: model_type = f.read().strip() # Get type
        if model_type == "keras": model = tf.keras.models.load_model(ARTIFACTS_DIR / "best_model_unified.keras") # Load DL
        else: model = load(ARTIFACTS_DIR / "best_model_unified.joblib") # Load ML
    except Exception as e: return print(f"Load Error: {e}. Run Module 01 & 02.") # Handle missing

    if "--batch" in sys.argv: batch_predict_from_db(preprocessor, model, model_type) # Batch
    else: predict_single(preprocessor, model, model_type) # Single

if __name__ == "__main__": main() # Start
