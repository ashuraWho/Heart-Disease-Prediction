# ============================================================ # Module 01 – EDA & Preprocessing
# NOTEBOOK 1 – EDA & PREPROCESSING                             # Heart Disease Prediction Dataset
# Heart Disease Prediction Dataset                             # Global Header
# ============================================================ # Global Header

# ===================== # Header Section
# 1. IMPORT LIBRARIES   # Import Libraries Header
# ===================== # Header Section

import os # Import os for environment variable manipulation
import sys # Import sys to check the Python environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda

# --- DIAGNOSTIC: Check Environment ---
print(f"Python Executable: {sys.executable}") # Print the path to the current Python interpreter
print(f"Python Version: {sys.version}") # Print the version of Python being used
if "anaconda3/bin/python" in sys.executable or "miniconda3/bin/python" in sys.executable: # Check if running in base
    print("WARNING: You are likely running in the 'base' environment. This is discouraged.") # Warn the user
# -------------------------------------

from pathlib import Path # Import Path for filesystem path manipulation

import pandas as pd                  # Import pandas for data manipulation and tabular data handling
import numpy as np                   # Import numpy for numerical computations and array operations

import matplotlib.pyplot as plt      # Import matplotlib for low-level plotting
import seaborn as sns                # Import seaborn for statistical data visualization

from sklearn.model_selection import train_test_split   # Import train_test_split for splitting data
from sklearn.preprocessing import StandardScaler       # Import StandardScaler for numerical feature scaling
from sklearn.preprocessing import OneHotEncoder        # Import OneHotEncoder for categorical feature encoding
from sklearn.compose import ColumnTransformer           # Import ColumnTransformer for selective preprocessing
from sklearn.pipeline import Pipeline                   # Import Pipeline for reproducible ML workflows

from joblib import dump # Import dump from joblib to save Python objects to disk

# ===================== # Header Section
# 2. GLOBAL CONFIG      # Global Configuration Header
# ===================== # Header Section

sns.set(style="whitegrid") # Set whitegrid style
plt.rcParams["figure.figsize"] = (10, 6) # Set default figure size

RANDOM_STATE = 42 # Set a random state for reproducibility
TEST_SIZE = 0.2 # Set the proportion of the dataset to include in the test split

# ===================== # Header Section
# 3. PATH HANDLING      # Path Handling Header
# ===================== # Header Section

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Define the root directory of the project
DATA_DIR = PROJECT_ROOT / "data" # Define the data directory path
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

DATASET_PATH = DATA_DIR / "heart_disease.csv" # Define the path to the dataset CSV file

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) # Create the artifacts directory if it doesn't exist

# ===================== # Header Section
# 4. LOAD DATASET       # Load Dataset Header
# ===================== # Header Section

try: # Start error handling block
    df = pd.read_csv(DATASET_PATH) # Load the dataset into a pandas DataFrame
except FileNotFoundError: # Catch missing file error
    print(f"ERROR: Dataset not found at {DATASET_PATH}") # Print error
    sys.exit(1) # Exit

print(f"Dataset loaded from: {DATASET_PATH}") # Print the dataset source path

# ===================== # Header Section
# 5. DATA CLEANING & NAN REMOVAL # Cleaning Header
# ===================== # Header Section

print(f"Initial row count: {df.shape[0]}") # Print initial count
print(f"Missing values per column:\n{df.isnull().sum()}") # Show NaNs
df.dropna(inplace=True) # DROP ALL NANs: This ensures the analysis is based on complete records
print(f"Row count after removing NaNs: {df.shape[0]}") # Print final count

# ===================== # Header Section
# 6. EXPLICIT CATEGORICAL MAPPING # Mapping Header
# ===================== # Header Section
# We map strings to numbers to enable correlation analysis and visualization.

# Mapping binary flags (Yes/No) to 1/0                                  # Binary mapping
binary_cols = [
    "Smoking", "Family Heart Disease", "Diabetes",
    "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol"
] # Define binary columns

for col in binary_cols: # Iterate through binary columns
    if col in df.columns: # Check if column exists
        df[col] = df[col].map({"Yes": 1, "No": 0}) # Map Yes to 1 and No to 0

# Mapping Gender                                                        # Gender mapping
if "Gender" in df.columns: # Check if Gender exists
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}) # Map Male to 1 and Female to 0

# Mapping Ordinal categories                                            # Ordinal mapping
# Level mapping: None: 0, Low: 1, Medium: 2, High: 3 (where applicable)
level_mapping = {"None": 0, "Low": 1, "Medium": 2, "High": 3} # Define mapping dict
level_mapping_basic = {"Low": 0, "Medium": 1, "High": 2} # Basic level mapping

if "Exercise Habits" in df.columns: # Check column
    df["Exercise Habits"] = df["Exercise Habits"].map(level_mapping_basic) # Map exercise
if "Alcohol Consumption" in df.columns: # Check column
    df["Alcohol Consumption"] = df["Alcohol Consumption"].map(level_mapping) # Map alcohol
if "Stress Level" in df.columns: # Check column
    df["Stress Level"] = df["Stress Level"].map(level_mapping_basic) # Map stress
if "Sugar Consumption" in df.columns: # Check column
    df["Sugar Consumption"] = df["Sugar Consumption"].map(level_mapping_basic) # Map sugar

# Mapping Target                                                        # Target mapping
if "Heart Disease Status" in df.columns: # Check column
    df["HeartDisease"] = df["Heart Disease Status"].map({"Yes": 1, "No": 0}) # Map Yes: 1, No: 0
    df.drop(columns=["Heart Disease Status"], inplace=True) # Drop original column name

print("\n--- DATA AFTER MAPPING ---") # Print header
print(df.head()) # Show mapped data
print(df.info()) # Show schema

# ===================== # Header Section
# 7. INTERACTIVE EDA FUNCTIONS # Plotting Header
# ===================== # Header Section

def show_correlation_matrix(): # Define matrix plot function
    plt.figure(figsize=(12, 10)) # Set size
    corr = df.corr() # Calculate correlations
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f") # Plot heatmap
    plt.title("Numerical Correlation Matrix (Including Mapped Categories)") # Set title
    plt.show() # Display

def show_target_distribution(): # Define target plot function
    sns.countplot(x="HeartDisease", data=df) # Plot counts
    plt.title("Distribution of Heart Disease Presence (Target)") # Set title
    plt.show() # Display

def show_feature_plot(feature_name): # Define generic feature plot function
    plt.figure(figsize=(14, 6)) # Set size
    plt.subplot(1, 2, 1) # Create subplot 1
    if feature_name in ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]: # Check if continuous
        sns.histplot(df[feature_name], kde=True) # Plot histogram
        plt.title(f"Distribution of {feature_name}") # Set title
    else: # If categorical
        sns.countplot(x=feature_name, data=df) # Plot count
        plt.title(f"Counts of {feature_name}") # Set title

    plt.subplot(1, 2, 2) # Create subplot 2
    sns.boxplot(x="HeartDisease", y=feature_name, data=df) # Plot boxplot vs target
    plt.title(f"{feature_name} vs Heart Disease") # Set title
    plt.show() # Display

# ===================== # Header Section
# 8. FEATURE / TARGET SPLIT # Data Split Header
# ===================== # Header Section

X = df.drop("HeartDisease", axis=1) # Create feature matrix
y = df["HeartDisease"] # Create target vector

# ===================== # Header Section
# 9. TRAIN / TEST SPLIT # Split Header
# ===================== # Header Section

X_train, X_test, y_train, y_test = train_test_split( # Split data
    X, # Features
    y, # Target
    test_size=TEST_SIZE, # Test size
    random_state=RANDOM_STATE, # Reproducibility
    stratify=y # Balance classes
) # End split

# ===================== # Header Section
# 10. PREPROCESSING PIPELINE # Pipeline Header
# ===================== # Header Section
# Since we mapped most categories manually, we primarily need scaling.

numerical_cols = X.columns.tolist() # Get all feature columns (they are all numeric now)

preprocessor = ColumnTransformer( # Define transformer
    transformers=[ # List transformers
        ("num", StandardScaler(), numerical_cols) # Scale everything (all mapped to numbers)
    ] # End list
) # End transformer

# ===================== # Header Section
# 11. FIT & TRANSFORM   # Fit Header
# ===================== # Header Section

X_train_processed = preprocessor.fit_transform(X_train) # Fit and transform train
X_test_processed = preprocessor.transform(X_test) # Transform test

# ===================== # Header Section
# 12. SAVE ARTIFACTS    # Save Header
# ===================== # Header Section

dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib") # Save preprocessor
np.save(ARTIFACTS_DIR / "y_train.npy", y_train) # Save labels
np.save(ARTIFACTS_DIR / "y_test.npy", y_test) # Save labels
np.savez(ARTIFACTS_DIR / "X_train.npz", X=X_train_processed) # Save features
np.savez(ARTIFACTS_DIR / "X_test.npz", X=X_test_processed) # Save features

print(f"\nArtifacts saved to: {ARTIFACTS_DIR.resolve()}") # Print path

# ===================== # Header Section
# 13. MENU INTERFACE    # CLI Interface Header
# ===================== # Header Section
# This allows showing plots one-by-one as requested.

def eda_menu(): # Define menu function
    while True: # Loop
        print("\n--- EDA PLOTTING MENU ---") # Print header
        print("1. Show Correlation Matrix") # Option 1
        print("2. Show Target Distribution") # Option 2
        print("3. Show individual feature plots") # Option 3
        print("q. Exit EDA plotting") # Exit
        choice = input("Select a plot (1-3 or q): ").lower() # Get input

        if choice == '1': show_correlation_matrix() # Handle choice 1
        elif choice == '2': show_target_distribution() # Handle choice 2
        elif choice == '3': # Handle choice 3
            for col in X.columns: # Iterate columns
                show_feature_plot(col) # Show plot
                if input("Press Enter for next feature, or 'q' to stop: ").lower() == 'q': break # Check break
        elif choice == 'q': break # Handle choice q
        else: print("Invalid selection.") # Handle invalid

if __name__ == "__main__": # Check execution
    if "--plots" in sys.argv: # If plots requested
        eda_menu() # Run menu
    else: # Default
        print("\n[INFO] Running preprocessing only. Use --plots for interactive graphs.") # Print notice
