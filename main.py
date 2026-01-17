# ============================================================ # Heart Disease Prediction - Integrated CLI Dashboard
# MAIN MODULE – MEDICAL DECISION SUPPORT SYSTEM                # Global Header
# Heart Disease Prediction                                     # Project Name
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
import sys # Import sys for system-specific parameters and functions
import subprocess # Import subprocess to run external scripts
from pathlib import Path # Import Path for robust filesystem path manipulation

# --- DIAGNOSTIC & STABILITY ---                                        # Stability Section
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN to improve stability
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads to prevent resource-related SegFaults
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU execution for stability on Mac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TensorFlow logging noise

# Define project paths                                                  # Path Definition Section
PROJECT_ROOT = Path(__file__).resolve().parent # Identify the project root directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks" # Define the directory containing modules
# ------------------------------

def clear_screen(): # Define a function to clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear') # Execute clear command based on OS

def print_header(title): # Define a function to print a styled header
    print("=" * 60) # Print decorative line
    print(f" {title.center(58)} ") # Print centered title
    print("=" * 60) # Print decorative line

def run_module(script_name): # Define a function to run a specific module script
    script_path = NOTEBOOKS_DIR / script_name # Construct the full path to the script
    print(f"\n[EXECUTION] Starting {script_name}...") # Print execution start message
    try: # Start error handling block
        subprocess.run([sys.executable, str(script_path)], check=True) # Run the script using the current Python interpreter
        print(f"\n[SUCCESS] {script_name} completed successfully.") # Print success message
    except subprocess.CalledProcessError as e: # Catch script execution errors
        print(f"\n[ERROR] Module {script_name} failed with exit code {e.returncode}.") # Print error message

def show_clinical_guide(): # Define a function to display clinical parameter explanations
    clear_screen() # Clear the terminal
    print_header("CLINICAL DATA GLOSSARY & GUIDE") # Print glossary header
    guide = { # Define dictionary with clinical parameter explanations
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
    } # End of guide dictionary
    for key, value in guide.items(): # Iterate through the guide dictionary
        print(f"► {key.ljust(15)}: {value}") # Print formatted explanation
    input("\nPress Enter to return to main menu...") # Wait for user input to continue

def reset_artifacts(): # Define a function to clear all generated artifacts
    if ARTIFACTS_DIR.exists(): # Check if directory exists
        import shutil # Import shutil for folder deletion
        shutil.rmtree(ARTIFACTS_DIR) # Delete the artifacts folder and its contents
        ARTIFACTS_DIR.mkdir() # Recreate the empty artifacts folder
        print("\n[RESET] All artifacts have been deleted. You MUST run the pipeline from Module 01.") # Print status
    else: # If directory doesn't exist
        print("\n[INFO] No artifacts found to delete.") # Print info

def main_menu(): # Define the main menu loop
    while True: # Start infinite loop for the menu
        clear_screen() # Clear the terminal
        print_header("HEART DISEASE PREDICTION SYSTEM - CLINICAL DASHBOARD") # Print system header
        print("1. [Pipeline] Run EDA & Preprocessing (Module 01)") # Option 1
        print("2. [Pipeline] Train & Tune Classical ML Models (Module 02)") # Option 2
        print("3. [XAI] View Model Explainability / SHAP Analysis (Module 03)") # Option 3
        print("4. [Deep Learning] Train Neural Network (Module 04)") # Option 4
        print("5. [Prediction] Predict for a New Patient (Module 05)") # Option 5
        print("6. [Knowledge] Clinical Data Glossary") # Option 6
        print("r. [Maintenance] Reset System Artifacts") # Reset option
        print("q. Exit System") # Exit option
        print("-" * 60) # Print separator
        print("NOTE: If you see 'AttributeError' or 'Compatibility Issue', run 1 and 2.") # Helpful tip
        print("-" * 60) # Print separator

        choice = input("Select an option (1-6, r or q): ").lower() # Get user choice and convert to lowercase

        match choice: # Use match-case for structural navigation
            case '1': # Case for EDA
                run_module("01_EDA_Preprocessing.py") # Run Module 01
                input("\nPress Enter to continue...") # Pause
            case '2': # Case for Classic ML
                run_module("02_ML_Classic.py") # Run Module 02
                input("\nPress Enter to continue...") # Pause
            case '3': # Case for Explainability
                run_module("03_Explainability.py") # Run Module 03
                input("\nPress Enter to continue...") # Pause
            case '4': # Case for Deep Learning
                run_module("04_Deep_Learning.py") # Run Module 04
                input("\nPress Enter to continue...") # Pause
            case '5': # Case for Inference
                run_module("05_Inference.py") # Run Module 05
                input("\nPress Enter to continue...") # Pause
            case '6': # Case for Glossary
                show_clinical_guide() # Show the glossary
            case 'r': # Case for Reset
                reset_artifacts() # Call reset function
                input("\nPress Enter to continue...") # Pause
            case 'q': # Case for Exit
                print("Exiting system. Stay healthy!") # Print goodbye message
                break # Exit the loop
            case _: # Default case for invalid input
                print("Invalid selection. Please try again.") # Print error
                input("\nPress Enter to continue...") # Pause

if __name__ == "__main__": # Check if the script is run directly
    main_menu() # Start the main menu
