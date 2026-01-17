# ============================================================ # Module 05 – Inference, Database & Clinical Support
# NOTEBOOK 5 – INTERACTIVE PREDICTION & SQL STORAGE           # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
import sys # Import sys to check the Python environment
import sqlite3 # Import sqlite3 for database management
import pandas as pd # Import pandas for data handling
import numpy as np # Import numpy for numerical operations
from pathlib import Path # Import Path for robust filesystem path manipulation
from joblib import load # Import load from joblib to retrieve saved models

# --- DIAGNOSTIC & STABILITY ---                                        # Stability Section
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda
# ------------------------------

# Define project paths                                                  # Path Definition Section
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Identify the project root directory
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path
DB_PATH = PROJECT_ROOT / "patients_data.db" # Define the path for the SQL database

# Clinical Glossary and Mapping Logic                                   # Mapping Section
# These must match the mappings used in Module 01.

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

# Mapping dictionaries for conversion                                   # Conversion Dictionaries
MAPPINGS = {
    "Binary": {"Yes": 1, "No": 0, "Male": 1, "Female": 0},
    "Ordinal_Basic": {"Low": 0, "Medium": 1, "High": 2},
    "Ordinal_Alcohol": {"None": 0, "Low": 1, "Medium": 2, "High": 3}
} # End mapping dicts

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
    ''') # Create table
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
    raw_data = {} # To store string inputs for SQL
    numeric_data = {} # To store converted values for Model

    for key, explanation in CLINICAL_GUIDE.items(): # Iterate glossary
        print(f"\n[GUIDE] {key}: {explanation}") # Print glossary info as requested
        while True: # Validation loop
            val = input(f"Enter {key}: ").strip() # Get input
            if not val: # Check empty
                print("Input required.") # Print error
                continue # Retry

            raw_data[key] = [val] # Store raw for SQL

            # Conversion Logic                                          # Conversion Section
            try:
                if key in ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]:
                    numeric_data[key] = [float(val)] # Convert to float
                elif key in ["Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol", "Gender"]:
                    if val not in MAPPINGS["Binary"]: raise ValueError("Use Yes/No or Male/Female")
                    numeric_data[key] = [MAPPINGS["Binary"][val]] # Convert binary
                elif key == "Alcohol Consumption":
                    if val not in MAPPINGS["Ordinal_Alcohol"]: raise ValueError("Use None/Low/Medium/High")
                    numeric_data[key] = [MAPPINGS["Ordinal_Alcohol"][val]] # Convert alcohol
                else: # Habits, Stress, Sugar
                    if val not in MAPPINGS["Ordinal_Basic"]: raise ValueError("Use Low/Medium/High")
                    numeric_data[key] = [MAPPINGS["Ordinal_Basic"][val]] # Convert ordinal
                break # Exit loop
            except ValueError as e: # Handle conversion error
                print(f"ERROR: {e}") # Print error

    return pd.DataFrame(raw_data), pd.DataFrame(numeric_data) # Return both

def predict_single(preprocessor, model): # Define function for single patient prediction flow
    raw_df, numeric_df = get_interactive_input() # Get data

    try: # Prediction block
        processed = preprocessor.transform(numeric_df) # Transform numeric data
        prediction = model.predict(processed)[0] # Predict
        prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else 0.0 # Prob

        print("\n--- PREDICTION RESULTS ---") # Print results
        print(f"RESULT: {'Heart Disease Detected' if prediction == 1 else 'No Disease detected'}") # Output
        print(f"Confidence: {prob:.2%}") # Output

        save_to_db(raw_df, prediction, prob) # Save raw strings to SQL for history
        print("\n[DB] Patient record saved.") # Confirmation
    except Exception as e: # Catch errors
        print(f"Prediction Error: {e}") # Output

def batch_predict_from_db(preprocessor, model): # Define batch function
    conn = sqlite3.connect(DB_PATH) # Connect
    try: # Block
        df = pd.read_sql_query("SELECT * FROM patients", conn) # Load
        if df.empty: # Check
            print("\nDatabase empty.") # Print
            return # Exit

        # We need to map strings in DB to numbers for re-prediction       # Re-mapping block
        df_numeric = df.copy().drop(columns=['id', 'Prediction', 'Probability', 'Timestamp']) # Drop meta
        for col in df_numeric.columns: # Iterate
            if col in ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]:
                df_numeric[col] = df_numeric[col].astype(float) # Ensure float
            elif col in ["Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol", "Gender"]:
                df_numeric[col] = df_numeric[col].map(MAPPINGS["Binary"]) # Map binary
            elif col == "Alcohol Consumption":
                df_numeric[col] = df_numeric[col].map(MAPPINGS["Ordinal_Alcohol"]) # Map alcohol
            else: # Basic ordinals
                df_numeric[col] = df_numeric[col].map(MAPPINGS["Ordinal_Basic"]) # Map ordinal

        processed = preprocessor.transform(df_numeric) # Transform
        df['New_Prediction'] = model.predict(processed) # Predict
        print("\n--- BATCH RESULTS ---") # Print
        print(df[['id', 'Age', 'Gender', 'Prediction', 'New_Prediction']]) # Show
    except Exception as e: # Catch
        print(f"Batch Error: {e}") # Output
    finally: # Close
        conn.close() # Close

def main(): # Local main
    init_db() # Init DB
    try: # Load artifacts
        if not (ARTIFACTS_DIR / "preprocessor.joblib").exists(): raise FileNotFoundError("preprocessor.joblib missing")
        preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib") # Load
        model = load(ARTIFACTS_DIR / "best_model_classic.joblib") # Load
    except Exception as e: # Catch
        print(f"Error loading artifacts: {e}") # Output
        return # Exit

    if "--batch" in sys.argv: batch_predict_from_db(preprocessor, model) # Batch mode
    else: predict_single(preprocessor, model) # Single mode

if __name__ == "__main__": main() # Run
