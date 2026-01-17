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

# Updated Clinical Glossary based on the NEW 21-column dataset structure # Glossary Section
CLINICAL_GUIDE = { # Define dictionary with clinical parameter explanations
    "Age": "The individual's age in years.", # Explanation for Age
    "Gender": "The individual's gender (Male or Female).", # Explanation for Gender
    "Blood Pressure": "The individual's systolic blood pressure.", # Explanation for BP
    "Cholesterol Level": "The individual's total cholesterol level.", # Explanation for Cholesterol
    "Exercise Habits": "The individual's exercise habits (Low, Medium, High).", # Explanation for Exercise
    "Smoking": "Whether the individual smokes or not (Yes or No).", # Explanation for Smoking
    "Family Heart Disease": "Whether there is a family history of heart disease (Yes or No).", # Explanation for Family History
    "Diabetes": "Whether the individual has diabetes (Yes or No).", # Explanation for Diabetes
    "BMI": "The individual's body mass index.", # Explanation for BMI
    "High Blood Pressure": "Whether the individual has high blood pressure (Yes or No).", # Explanation for High BP
    "Low HDL Cholesterol": "Whether the individual has low HDL cholesterol (Yes or No).", # Explanation for Low HDL
    "High LDL Cholesterol": "Whether the individual has high LDL cholesterol (Yes or No).", # Explanation for High LDL
    "Alcohol Consumption": "The individual's alcohol consumption level (None, Low, Medium, High).", # Explanation for Alcohol
    "Stress Level": "The individual's stress level (Low, Medium, High).", # Explanation for Stress
    "Sleep Hours": "The number of hours the individual sleeps.", # Explanation for Sleep
    "Sugar Consumption": "The individual's sugar consumption level (Low, Medium, High).", # Explanation for Sugar
    "Triglyceride Level": "The individual's triglyceride level.", # Explanation for Triglycerides
    "Fasting Blood Sugar": "The individual's fasting blood sugar level.", # Explanation for FBS
    "CRP Level": "The C-reactive protein level (marker of inflammation).", # Explanation for CRP
    "Homocysteine Level": "The individual's homocysteine level (affects vessel health)." # Explanation for Homocysteine
} # End of updated glossary

def init_db(): # Define function to initialize the SQL database
    conn = sqlite3.connect(DB_PATH) # Connect to the SQLite database
    cursor = conn.cursor() # Create a cursor object
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
    ''') # Execute SQL command to create the patients table with metadata for the new dataset
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
            val = input(f"Enter value for {key}: ") # Ask for input
            if val.strip() == "": # Check for empty input
                print("Input cannot be empty.") # Print error
                continue # Retry
            # We store everything as string/float as appropriate, pandas will handle it
            patient_dict[key] = [val] # Store the value
            break # Exit loop
    return pd.DataFrame(patient_dict) # Return the data as a pandas DataFrame

def predict_single(preprocessor, model): # Define function for single patient prediction flow
    new_patient_df = get_interactive_input() # Get interactive input from the user

    try: # Start safety block for preprocessing
        new_patient_processed = preprocessor.transform(new_patient_df) # Process the raw input data
        prediction = model.predict(new_patient_processed)[0] # Get the class prediction

        if hasattr(model, "predict_proba"): # Check if model can predict probabilities
            probability = model.predict_proba(new_patient_processed)[0] # Get probabilities
            conf = probability[1] # Confidence of presence
        else: # Fallback for models without predict_proba
            conf = 0.0 # Default confidence

        print("\n--- PREDICTION RESULTS ---") # Print results header
        result_text = "Presence of Heart Disease detected." if prediction == 1 else "No heart disease detected (Absence)." # Define result text
        print(f"RESULT: {result_text}") # Output result
        if hasattr(model, "predict_proba"): # Output confidence if available
            print(f"Confidence (Presence Probability): {conf:.2%}") # Output confidence

        save_to_db(new_patient_df, prediction, conf) # Save the patient record and results to SQL
        print("\n[DB] Patient record and prediction saved to SQL database.") # Print confirmation
    except Exception as e: # Catch errors during inference
        print(f"\n[ERROR] Prediction failed: {e}") # Print error message
        print(">>> Ensure the data entered matches the categories used during training.") # Print tip

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
        if hasattr(model, "predict_proba"): # Generate new probabilities if available
            df['New_Probability'] = model.predict_proba(X_processed)[:, 1] # Generate new probabilities

        print("\n--- BATCH PREDICTION RESULTS (FROM DATABASE) ---") # Print batch header
        cols_to_show = ['id', 'Age', 'Gender', 'Prediction', 'New_Prediction'] # Basic columns
        if 'New_Probability' in df.columns: # Add probability if present
             cols_to_show.append('New_Probability') # Append column
        print(df[cols_to_show]) # Display summary
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
