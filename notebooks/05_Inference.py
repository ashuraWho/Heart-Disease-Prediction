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

# Clinical Glossary for user guidance during input                      # Glossary Section
CLINICAL_GUIDE = { # Define dictionary with clinical parameter explanations
    "Age": "Patient's age in years.", # Explanation for Age
    "Sex": "1 = Male; 0 = Female.", # Explanation for Sex
    "ChestPain": "1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain, 4: Asymptomatic.", # Explanation for Chest Pain
    "BP": "Resting blood pressure (mm Hg on admission to the hospital).", # Explanation for Blood Pressure
    "Cholesterol": "Serum cholesterol in mg/dl.", # Explanation for Cholesterol
    "FBS": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).", # Explanation for Fasting Blood Sugar
    "EKG": "Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy).", # Explanation for ECG
    "MaxHR": "Maximum heart rate achieved during stress test.", # Explanation for Max Heart Rate
    "ExerciseAngina": "Exercise induced angina (1 = yes; 0 = no).", # Explanation for Exercise Angina
    "ST_Depression": "ST depression induced by exercise relative to rest (marker of ischemia).", # Explanation for ST Depression
    "ST_Slope": "1: Upsloping, 2: Flat, 3: Downsloping (Slope of the peak exercise ST segment).", # Explanation for ST Slope
    "NumVessels": "Number of major vessels (0-3) colored by flourosopy.", # Explanation for Number of Vessels
    "Thallium": "3 = Normal; 6 = Fixed defect; 7 = Reversable defect (Nuclear stress test result)." # Explanation for Thallium Test
} # End of glossary

def init_db(): # Define function to initialize the SQL database
    conn = sqlite3.connect(DB_PATH) # Connect to the SQLite database
    cursor = conn.cursor() # Create a cursor object
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Age REAL, Sex REAL, ChestPain REAL, BP REAL, Cholesterol REAL,
            FBS REAL, EKG REAL, MaxHR REAL, ExerciseAngina REAL,
            ST_Depression REAL, ST_Slope REAL, NumVessels REAL, Thallium REAL,
            Prediction INTEGER, Probability REAL, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''') # Execute SQL command to create the patients table with metadata
    conn.commit() # Commit the changes
    conn.close() # Close the connection

def save_to_db(data, prediction, probability): # Define function to save data to SQL
    conn = sqlite3.connect(DB_PATH) # Connect to the SQLite database
    df_to_save = data.copy() # Create a copy of the input data
    df_to_save['Prediction'] = int(prediction) # Add the prediction to the row
    df_to_save['Probability'] = float(probability) # Add the positive class probability to the row
    df_to_save.to_sql('patients', conn, if_exists='append', index=False) # Append the data to the SQL table
    conn.close() # Close the connection

def get_interactive_input(): # Define function for manual patient data entry
    print("\n--- NEW PATIENT DATA ENTRY ---") # Print entry header
    patient_dict = {} # Initialize empty dictionary for the patient
    for key, explanation in CLINICAL_GUIDE.items(): # Iterate through the glossary
        print(f"\n[INFO] {key}: {explanation}") # Print the explanation for the current parameter
        while True: # Start validation loop
            try: # Start error handling block
                val = float(input(f"Enter value for {key}: ")) # Ask for input and convert to float
                patient_dict[key] = [val] # Store the value in the dictionary
                break # Exit the validation loop if successful
            except ValueError: # Catch non-numeric inputs
                print("Invalid input. Please enter a numeric value.") # Print error message
    return pd.DataFrame(patient_dict) # Return the data as a pandas DataFrame

def predict_single(preprocessor, model): # Define function for single patient prediction flow
    new_patient_df = get_interactive_input() # Get interactive input from the user
    new_patient_processed = preprocessor.transform(new_patient_df) # Process the raw input data
    prediction = model.predict(new_patient_processed)[0] # Get the class prediction
    probability = model.predict_proba(new_patient_processed)[0] # Get the class probabilities

    print("\n--- PREDICTION RESULTS ---") # Print results header
    result_text = "Presence of Heart Disease detected." if prediction == 1 else "No heart disease detected (Absence)." # Define result text
    print(f"RESULT: {result_text}") # Output result
    print(f"Confidence (Presence Probability): {probability[1]:.2%}") # Output confidence

    save_to_db(new_patient_df, prediction, probability[1]) # Save the patient record and results to SQL
    print("\n[DB] Patient record and prediction saved to SQL database.") # Print confirmation

def batch_predict_from_db(preprocessor, model): # Define function to re-predict all stored records
    conn = sqlite3.connect(DB_PATH) # Connect to the SQLite database
    try: # Start error handling block
        df = pd.read_sql_query("SELECT * FROM patients", conn) # Load all data from the patients table
        if df.empty: # Check if the table is empty
            print("\n[INFO] Database is empty. No patients to predict.") # Print info message
            return # Exit function

        # Drop metadata columns before preprocessing
        X_raw = df.drop(columns=['id', 'Prediction', 'Probability', 'Timestamp']) # Extract raw clinical features
        X_processed = preprocessor.transform(X_raw) # Preprocess the data

        df['New_Prediction'] = model.predict(X_processed) # Generate new predictions
        df['New_Probability'] = model.predict_proba(X_processed)[:, 1] # Generate new probabilities

        print("\n--- BATCH PREDICTION RESULTS (FROM DATABASE) ---") # Print batch header
        print(df[['id', 'Age', 'Sex', 'Prediction', 'New_Prediction', 'New_Probability']]) # Display summary
    except Exception as e: # Catch any errors during database operation
        print(f"\n[ERROR] Could not perform batch prediction: {e}") # Print error message
    finally: # Ensure connection is closed
        conn.close() # Close the database connection

def main(): # Define the local main function for the script
    init_db() # Ensure the database and table exist

    # Load artifacts with version safety                                # Loading Section
    try: # Start safety block
        if not (ARTIFACTS_DIR / "preprocessor.joblib").exists() or not (ARTIFACTS_DIR / "best_model_classic.joblib").exists(): # Check artifacts
             print("ERROR: Artifacts missing. Run Module 01 & 02.") # Print error
             return # Exit
        preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib") # Load preprocessor
        model = load(ARTIFACTS_DIR / "best_model_classic.joblib") # Load model
    except Exception as e: # Catch loading errors
        print(f"ERROR loading artifacts: {e}") # Print error
        return # Exit

    if len(sys.argv) > 1 and sys.argv[1] == "--batch": # Check for batch mode command line argument
        batch_predict_from_db(preprocessor, model) # Execute batch prediction
    else: # Default to interactive entry
        predict_single(preprocessor, model) # Execute single prediction flow

if __name__ == "__main__": # Check if script is run directly
    main() # Call main function
