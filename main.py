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
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the directory for ML artifacts
DB_PATH = PROJECT_ROOT / "patients_data.db" # Define the path for the SQL database
# ------------------------------

def clear_screen(): # Define a function to clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear') # Execute clear command based on OS

def print_header(title): # Define a function to print a styled header
    print("=" * 60) # Print decorative line
    print(f" {title.center(58)} ") # Print centered title
    print("=" * 60) # Print decorative line

def run_module(script_name, args=None): # Define a function to run a specific module script
    script_path = NOTEBOOKS_DIR / script_name # Construct the full path to the script
    print(f"\n[EXECUTION] Starting {script_name}...") # Print execution start message
    cmd = [sys.executable, str(script_path)] # Prepare the base command
    if args: # Check if additional arguments are provided
        cmd.extend(args) # Add arguments to the command
    try: # Start error handling block
        subprocess.run(cmd, check=True) # Run the script using the current Python interpreter
        print(f"\n[SUCCESS] {script_name} completed successfully.") # Print success message
    except subprocess.CalledProcessError as e: # Catch script execution errors
        print(f"\n[ERROR] Module {script_name} failed with exit code {e.returncode}.") # Print error message

def show_clinical_guide(): # Define a function to display clinical parameter explanations
    clear_screen() # Clear the terminal
    print_header("CLINICAL DATA GLOSSARY & GUIDE") # Print glossary header
    guide = { # Define dictionary with clinical parameter explanations
        "Age": "Patient's age in years.", # Explanation for Age
        "Gender": "Patient's biological gender (Male/Female).", # Explanation for Gender
        "Blood Pressure": "Systolic blood pressure (mm Hg).", # Explanation for Blood Pressure
        "Cholesterol Level": "Total serum cholesterol level (mg/dl).", # Explanation for Cholesterol
        "Exercise Habits": "Level of regular physical activity (Low, Medium, High).", # Explanation for Exercise
        "Smoking": "Current smoking status (Yes/No).", # Explanation for Smoking
        "Family Heart Disease": "History of heart disease in immediate family (Yes/No).", # Explanation for Family History
        "Diabetes": "Whether the patient has a diabetes diagnosis (Yes/No).", # Explanation for Diabetes
        "BMI": "Body Mass Index (weight / height^2).", # Explanation for BMI
        "High Blood Pressure": "Pre-existing diagnosis of hypertension (Yes/No).", # Explanation for High BP
        "Low HDL Cholesterol": "Presence of low 'good' cholesterol (Yes/No).", # Explanation for Low HDL
        "High LDL Cholesterol": "Presence of high 'bad' cholesterol (Yes/No).", # Explanation for High LDL
        "Alcohol Consumption": "Level of alcohol intake (None, Low, Medium, High).", # Explanation for Alcohol
        "Stress Level": "Reported psychological stress level (Low, Medium, High).", # Explanation for Stress
        "Sleep Hours": "Average hours of sleep per night.", # Explanation for Sleep
        "Sugar Consumption": "Dietary sugar intake level (Low, Medium, High).", # Explanation for Sugar
        "Triglyceride Level": "Serum triglyceride level (mg/dl).", # Explanation for Triglycerides
        "Fasting Blood Sugar": "Blood sugar level after fasting.", # Explanation for Fasting Sugar
        "CRP Level": "C-reactive protein level (marker of inflammation).", # Explanation for CRP
        "Homocysteine Level": "Homocysteine level (marker for vascular health)." # Explanation for Homocysteine
    } # End of guide dictionary
    for key, value in guide.items(): # Iterate through the guide dictionary
        print(f"► {key.ljust(20)}: {value}") # Print formatted explanation
    input("\nPress Enter to return to main menu...") # Wait for user input to continue

def reset_artifacts(): # Define a function to clear all generated artifacts
    if ARTIFACTS_DIR.exists(): # Check if directory exists
        import shutil # Import shutil for folder deletion
        shutil.rmtree(ARTIFACTS_DIR) # Delete the artifacts folder and its contents
        ARTIFACTS_DIR.mkdir() # Recreate the empty artifacts folder
        print("\n[RESET] All artifacts have been deleted. You MUST run the pipeline from Module 01.") # Print status
    else: # If directory doesn't exist
        print("\n[INFO] No artifacts found to delete.") # Print info

def delete_database(): # Define function to delete the SQL database
    if DB_PATH.exists(): # Check if database file exists
        os.remove(DB_PATH) # Delete the file
        print(f"\n[DELETE] Database '{DB_PATH.name}' has been successfully deleted.") # Print status
    else: # If file doesn't exist
        print("\n[INFO] No database found to delete.") # Print info

def main_menu(): # Define the main menu loop
    while True: # Start infinite loop for the menu
        clear_screen() # Clear the terminal
        print_header("HEART DISEASE PREDICTION SYSTEM - CLINICAL DASHBOARD") # Print system header
        print("1. [Pipeline] Run EDA & Preprocessing (Module 01)") # Option 1
        print("2. [Pipeline] Train & Tune Classical ML Models (Module 02)") # Option 2
        print("3. [XAI] View Model Explainability / SHAP Analysis (Module 03)") # Option 3
        print("4. [Deep Learning] Train Neural Network (Module 04)") # Option 4
        print("5. [Prediction] Predict for a New Patient (Manual Input & Save)") # Option 5
        print("6. [Prediction] Batch Predict all Patients in Database") # Option 6
        print("7. [Knowledge] Clinical Data Glossary") # Option 7
        print("d. [Maintenance] Delete SQL Database") # Delete DB option
        print("r. [Maintenance] Reset System Artifacts") # Reset option
        print("q. Exit System") # Exit option
        print("-" * 60) # Print separator
        print("NOTE: If you see 'AttributeError' or 'Compatibility Issue', run 1 and 2.") # Helpful tip
        print("-" * 60) # Print separator

        choice = input("Select an option (1-7, d, r or q): ").lower() # Get user choice and convert to lowercase

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
            case '6': # Case for Batch Prediction
                run_module("05_Inference.py", args=["--batch"]) # Run Module 05 in batch mode
                input("\nPress Enter to continue...") # Pause
            case '7': # Case for Glossary
                show_clinical_guide() # Show the glossary
            case 'd': # Case for Delete DB
                delete_database() # Call delete DB function
                input("\nPress Enter to continue...") # Pause
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
