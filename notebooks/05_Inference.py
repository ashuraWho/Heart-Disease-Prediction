# ============================================================ # Module 05 – Inference & Prediction
# NOTEBOOK 5 – REAL-WORLD INFERENCE                           # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
import sys # Import sys to check the Python environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda

# --- DIAGNOSTIC: Check Environment ---
print(f"Python Executable: {sys.executable}") # Print the path to the current Python interpreter
print(f"Python Version: {sys.version}") # Print the version of Python being used
if "anaconda3/bin/python" in sys.executable or "miniconda3/bin/python" in sys.executable: # Check if running in base
    print("WARNING: You are likely running in the 'base' environment. This is discouraged.") # Warn the user
# -------------------------------------

from pathlib import Path # Import Path for robust filesystem path manipulation
import pandas as pd # Import pandas for data manipulation
import numpy as np # Import numpy for numerical operations
from joblib import load # Import load from joblib to retrieve saved models

# ===================== # Header Section
# PATHS                 # Paths Header
# ===================== # Header Section

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Define the root directory of the project
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

# ===================== # Header Section
# LOAD ARTIFACTS        # Load Artifacts Header
# ===================== # Header Section

# Check if artifacts exist before loading                               # Artifact Check Comment
if not (ARTIFACTS_DIR / "preprocessor.joblib").exists(): # Verify preprocessor exists
    print("ERROR: preprocessor.joblib not found. Please run Module 01 first.") # Print error message
    sys.exit(1) # Exit script if missing

if not (ARTIFACTS_DIR / "best_model_classic.joblib").exists(): # Verify model exists
    print("ERROR: best_model_classic.joblib not found. Please run Module 02 first.") # Print error message
    sys.exit(1) # Exit script if missing

preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib") # Load the saved preprocessing pipeline
model = load(ARTIFACTS_DIR / "best_model_classic.joblib") # Load the best classical model (e.g., KNN, RF, or LR)

print("Inference artifacts loaded successfully.") # Print confirmation message

# ===================== # Header Section
# SAMPLE PATIENT DATA   # Sample Data Header
# ===================== # Header Section
# This dictionary represents a new patient coming into the clinic.     # Sample patient comment
# Ensure the keys match the column names expected by the preprocessor.   # Key matching comment

new_patient_data = { # Define a dictionary with clinical parameters
    "Age": [65], # Patient age in years
    "Sex": [1], # Sex (1 = male; 0 = female)
    "ChestPain": [4], # Chest pain type (1-4)
    "BP": [140], # Resting blood pressure
    "Cholesterol": [240], # Serum cholesterol in mg/dl
    "FBS": [0], # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "EKG": [2], # Resting electrocardiographic results (0-2)
    "MaxHR": [150], # Maximum heart rate achieved
    "ExerciseAngina": [0], # Exercise induced angina (1 = yes; 0 = no)
    "ST_Depression": [2.3], # ST depression induced by exercise relative to rest
    "ST_Slope": [2], # The slope of the peak exercise ST segment (1-3)
    "NumVessels": [1], # Number of major vessels (0-3) colored by flourosopy
    "Thallium": [7] # Thallium stress test (3 = normal; 6 = fixed defect; 7 = reversable defect)
} # End of patient dictionary

# Convert dictionary to DataFrame                                       # Conversion Comment
new_patient_df = pd.DataFrame(new_patient_data) # Create a pandas DataFrame from the patient data

print("\n--- NEW PATIENT CLINICAL DATA ---") # Print clinical data header
print(new_patient_df) # Display the input data

# ===================== # Header Section
# PREPROCESSING         # Preprocessing Header
# ===================== # Header Section

# Transform the raw data using the fitted preprocessor                  # Transformation Comment
new_patient_processed = preprocessor.transform(new_patient_df) # Apply scaling and encoding to the new data

# ===================== # Header Section
# PREDICTION            # Prediction Header
# ===================== # Header Section

# Make the final prediction                                             # Prediction Comment
prediction = model.predict(new_patient_processed)[0] # Predict class (0 = Absence, 1 = Presence)
probability = model.predict_proba(new_patient_processed)[0] # Predict class probabilities

# ===================== # Header Section
# RESULTS OUTPUT        # Results Output Header
# ===================== # Header Section

print("\n--- PREDICTION RESULTS ---") # Print results header
if prediction == 1: # Check if heart disease is predicted
    print("RESULT: Presence of Heart Disease detected.") # Print positive result
else: # If heart disease is not predicted
    print("RESULT: No heart disease detected (Absence).") # Print negative result

print(f"Confidence (Probability):") # Print probability header
print(f" - Probability of Absence: {probability[0]:.2%}") # Print probability of no disease
print(f" - Probability of Presence: {probability[1]:.2%}") # Print probability of disease

print("\n--- MEDICAL NOTE ---") # Print medical note header
print("This tool is for educational and research support only.") # Print legal disclaimer
print("Final diagnosis must always be confirmed by a professional cardiologist.") # Print professional recommendation
