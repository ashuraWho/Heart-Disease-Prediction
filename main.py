# ============================================================ # Heart Disease Prediction - Integrated CLI Dashboard
# MAIN MODULE – MEDICAL DECISION SUPPORT SYSTEM                # Global Header
# Heart Disease Prediction                                     # Project Name
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
# --- CRITICAL STABILITY & macOS FIXES (MUST BE BEFORE ANY OTHER IMPORTS) ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN to improve stability
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1' # Limit MKL threads
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce logging
# ---------------------------------------------------------------------------

import sys # Import sys for system-specific parameters and functions
import subprocess # Import subprocess to run external scripts
from pathlib import Path # Import Path for robust filesystem path manipulation

# Define project paths                                                  # Path Definition Section
PROJECT_ROOT = Path(__file__).resolve().parent # Identify the project root directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks" # Define the directory containing modules
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the directory for ML artifacts
DB_PATH = PROJECT_ROOT / "patients_data.db" # Define the path for the SQL database
# ------------------------------

def clear_screen(): # Define a function to clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear') # Execute clear command based on OS

def print_header(title): # Define a function to print a styled header
    print("\n" + "=" * 60) # Print decorative line
    print(f" {title.center(58)} ") # Print centered title
    print("=" * 60) # Print decorative line

def run_module(script_name, args=None): # Define a function to run a specific module script
    script_path = NOTEBOOKS_DIR / script_name # Construct the full path to the script
    print(f"\n[EXECUTION] Starting {script_name}...") # Print execution start message

    # Pass current environment variables to the subprocess                  # Subprocess Env handling
    env = os.environ.copy() # Copy current env

    cmd = [sys.executable, str(script_path)] # Prepare the base command
    if args: # Check if additional arguments are provided
        cmd.extend(args) # Add arguments to the command
    try: # Start error handling block
        subprocess.run(cmd, check=True, env=env) # Run the script with inherited environment
        print(f"\n[SUCCESS] {script_name} completed successfully.") # Print success message
    except subprocess.CalledProcessError as e: # Catch script execution errors
        if e.returncode == -11 or e.returncode == 139: # Check for SegFault codes
             print(f"\n[CRITICAL ERROR] Module {script_name} experienced a Segmentation Fault (Mac/Anaconda conflict).")
             print(">>> Please ensure you are NOT in the 'base' environment and use 'setup_mac.sh'.")
        else:
             print(f"\n[ERROR] Module {script_name} failed with exit code {e.returncode}.") # Print error message

def show_clinical_guide(): # Define a function to display clinical parameter explanations
    clear_screen() # Clear the terminal
    print_header("CLINICAL DATA GLOSSARY & GUIDE") # Print glossary header
    guide = { # Define dictionary with clinical parameter explanations
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
    } # End of guide dictionary
    for key, value in guide.items(): # Iterate through the guide dictionary
        print(f"► {key.ljust(20)}: {value}") # Print formatted explanation
    input("\nPress Enter to return to main menu...") # Wait for user input to continue

def eda_interactive_menu(): # Define sub-menu for EDA plots
    while True: # Infinite loop
        clear_screen() # Clear
        print_header("EDA & DATA VISUALIZATION") # Header
        print("1. Show Correlation Matrix (Numerical Heatmap)") # Option 1
        print("2. Show Target Variable Distribution") # Option 2
        print("3. Show Individual Feature Plots (One by One)") # Option 3
        print("q. Return to Main Menu") # Return

        choice = input("\nSelect an option: ").lower() # Get choice
        match choice: # Match-case
            case '1': run_module("01_EDA_Preprocessing.py", args=["--plots"]) # Pass arg
            case '2': run_module("01_EDA_Preprocessing.py", args=["--plots"]) # Pass arg
            case '3': run_module("01_EDA_Preprocessing.py", args=["--plots"]) # Pass arg
            case 'q': break # Exit sub-menu
            case _: print("Invalid selection.")

def reset_artifacts(): # Define a function to clear all generated artifacts
    if ARTIFACTS_DIR.exists(): # Check if directory exists
        import shutil # Import shutil for folder deletion
        shutil.rmtree(ARTIFACTS_DIR) # Delete the artifacts folder and its contents
        ARTIFACTS_DIR.mkdir() # Recreate the empty artifacts folder
        print("\n[RESET] All artifacts deleted. You MUST run Module 01 & 02.") # Status
    else: # If directory doesn't exist
        print("\n[INFO] No artifacts folder found.") # Info

def delete_database(): # Define function to delete the SQL database
    if DB_PATH.exists(): # Check if database file exists
        os.remove(DB_PATH) # Delete the file
        print(f"\n[DELETE] Database '{DB_PATH.name}' deleted.") # Status
    else: # If file doesn't exist
        print("\n[INFO] No database found.") # Info

def main_menu(): # Define the main menu loop
    while True: # Start infinite loop for the menu
        clear_screen() # Clear the terminal
        print_header("HEART DISEASE PREDICTION SYSTEM - CLINICAL DASHBOARD") # Print system header
        print("1. [Data] EDA & Visual Analysis (Interactive Plots)") # Option 1
        print("2. [Training] Unified Model Competition (ML vs DL)") # Option 2
        print("3. [XAI] Explainability (SHAP Analysis) - Why the prediction?") # Option 3
        print("4. [Patient] Predict for a New Patient (Manual Entry & SQL Save)") # Option 4
        print("5. [History] Batch Predict all Patients in Database") # Option 5
        print("6. [Knowledge] Clinical Glossary (Definitions)") # Option 6
        print("d. [Maintenance] Delete SQL Patient Database") # Delete DB option
        print("r. [Maintenance] Reset ML Artifacts (Regenerate Models)") # Reset option
        print("q. Exit System") # Exit option
        print("-" * 60) # Print separator
        print("Tip: Run Module 1 & 2 first to enable Predictions and XAI.") # Helpful tip

        choice = input("\nSelect an option (1-6, d, r or q): ").lower() # Get user choice

        match choice: # Use match-case for structural navigation
            case '1': eda_interactive_menu() # Run EDA sub-menu
            case '2': # Unified Training
                run_module("02_Unified_Training.py") # Run new module
                input("\nPress Enter...") # Pause
            case '3': # Explainability
                run_module("03_Explainability.py") # Run
                input("\nPress Enter...") # Pause
            case '4': # Inference
                run_module("04_Inference.py") # Run
                input("\nPress Enter...") # Pause
            case '5': # Batch Prediction
                run_module("04_Inference.py", args=["--batch"]) # Run in batch
                input("\nPress Enter...") # Pause
            case '6': # Glossary
                show_clinical_guide() # Show
            case 'd': # Delete DB
                delete_database() # Call
                input("\nPress Enter...") # Pause
            case 'r': # Reset
                reset_artifacts() # Call
                input("\nPress Enter...") # Pause
            case 'q': # Exit
                print("Exiting. Stay healthy!") # Goodbye
                break # Exit
            case _: # Invalid
                print("Invalid choice.") # Error
                input("\nPress Enter...") # Pause

if __name__ == "__main__": # Entry
    main_menu() # Start
