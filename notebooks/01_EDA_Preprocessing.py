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

DATASET_PATH = DATA_DIR / "heart_disease.csv" # Define the path to the NEW dataset CSV file

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) # Create the artifacts directory if it doesn't exist

# ===================== # Header Section
# 4. LOAD DATASET       # Load Dataset Header
# ===================== # Header Section

try: # Start error handling block
    df = pd.read_csv(DATASET_PATH) # Load the dataset into a pandas DataFrame
except FileNotFoundError: # Catch missing file error
    print(f"ERROR: Dataset not found at {DATASET_PATH}") # Print error
    print(">>> Please ensure 'heart_disease.csv' is in the 'data/' folder.") # Print fix
    sys.exit(1) # Exit

print(f"Dataset loaded from: {DATASET_PATH}") # Print the dataset source path
print(df.head()) # Print the first five rows of the DataFrame

# ===================== # Header Section
# 5. DATASET OVERVIEW & CLEANING # Overview and Cleaning Header
# ===================== # Header Section

print(f"Number of rows before cleaning: {df.shape[0]}") # Print row count
print(f"Number of columns: {df.shape[1]}") # Print column count

# DROPPING MISSING VALUES                                               # Drop missing values requirement
df.dropna(inplace=True) # Remove all rows containing at least one missing value
print(f"Number of rows after dropping missing values: {df.shape[0]}") # Print updated row count

df.info() # Print a concise summary of the DataFrame

# ===================== # Header Section
# 6. TARGET ENCODING    # Target Encoding Header
# ===================== # Header Section

# Target column name: "Heart Disease Status"                            # New target name
if "Heart Disease Status" in df.columns: # Check if column exists
    df["HeartDisease"] = df["Heart Disease Status"].map({ # Map target labels to numerical values
        "No": 0, # Map 'No' heart disease to 0
        "Yes": 1 # Map 'Yes' heart disease to 1
    }) # End of mapping
    df.drop(columns=["Heart Disease Status"], inplace=True) # Drop original target column
else: # Fallback
    print("ERROR: Target column 'Heart Disease Status' not found.") # Print error
    sys.exit(1) # Exit

# ===================== # Header Section
# 7. DATA QUALITY CHECKS # Data Quality Checks Header
# ===================== # Header Section

duplicates = df.duplicated().sum() # Count the number of duplicate rows in the DataFrame
print(f"\nNumber of duplicate rows: {duplicates}") # Print the count of duplicate rows

# ===================== # Header Section
# 8. TARGET ANALYSIS    # Target Analysis Header
# ===================== # Header Section

sns.countplot(x="HeartDisease", data=df) # Create a count plot for the target variable
plt.title("Target Distribution (HeartDisease)") # Set the title of the target distribution plot
plt.show() # Display the target distribution plot

# ===================== # Header Section
# 9. FEATURE GROUPS     # Feature Groups Header
# ===================== # Header Section

# Define features based on the NEW 21-column dataset structure         # New feature groups
numerical_features = [ # Define a list of numerical features
    "Age", "Blood Pressure", "Cholesterol Level", "BMI",
    "Sleep Hours", "Triglyceride Level", "Fasting Blood Sugar",
    "CRP Level", "Homocysteine Level"
] # End of numerical features list

categorical_features = [ # Define a list of categorical features
    "Gender", "Exercise Habits", "Smoking", "Family Heart Disease",
    "Diabetes", "High Blood Pressure", "Low HDL Cholesterol",
    "High LDL Cholesterol", "Alcohol Consumption", "Stress Level",
    "Sugar Consumption"
] # End of categorical features list

# Ensure columns exist in df before defining the pipeline               # Safety check
numerical_features = [f for f in numerical_features if f in df.columns] # Filter existing numeric
categorical_features = [f for f in categorical_features if f in df.columns] # Filter existing categorical

# ===================== # Header Section
# 10. CORRELATION ANALYSIS # Correlation Analysis Header
# ===================== # Header Section

corr_matrix = df[numerical_features + ["HeartDisease"]].corr() # Calculate the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm") # Create a heatmap
plt.title("Correlation Matrix (Numerical Features)") # Set title
plt.show() # Display

# ===================== # Header Section
# 11. FEATURE / TARGET SPLIT # Data Split Header
# ===================== # Header Section

X = df.drop("HeartDisease", axis=1) # Create the feature matrix
y = df["HeartDisease"] # Create the target vector

# ===================== # Header Section
# 12. TRAIN / TEST SPLIT # Train/Test Split Header
# ===================== # Header Section

X_train, X_test, y_train, y_test = train_test_split( # Split data
    X, # Feature matrix
    y, # Target vector
    test_size=TEST_SIZE, # Proportion of test set
    random_state=RANDOM_STATE, # Random state
    stratify=y # Stratify split
) # End of split function

# ===================== # Header Section
# 13. PREPROCESSING PIPELINE # Preprocessing Pipeline Header
# ===================== # Header Section

numeric_transformer = Pipeline(steps=[ # Define numeric pipeline
    ("scaler", StandardScaler()) # Scale features
]) # End of numeric pipeline

categorical_transformer = Pipeline(steps=[ # Define categorical pipeline
    ("onehot", OneHotEncoder( # Encode features
        drop="first", # Drop first to avoid collinearity
        handle_unknown="ignore" # Ignore unknown
    )) # End of OneHotEncoder
]) # End of categorical pipeline

preprocessor = ColumnTransformer( # Combine transformations
    transformers=[ # List of transformers
        ("num", numeric_transformer, numerical_features), # Apply numeric
        ("cat", categorical_transformer, categorical_features) # Apply categorical
    ] # End of transformer list
) # End of ColumnTransformer

# ===================== # Header Section
# 14. FIT & TRANSFORM   # Fit and Transform Header
# ===================== # Header Section

X_train_processed = preprocessor.fit_transform(X_train) # Fit and transform training
X_test_processed = preprocessor.transform(X_test) # Transform testing

# ===================== # Header Section
# 15. SAVE ARTIFACTS    # Save Artifacts Header
# ===================== # Header Section

dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib") # Save the fitted preprocessor

np.save(ARTIFACTS_DIR / "y_train.npy", y_train) # Save training labels
np.save(ARTIFACTS_DIR / "y_test.npy", y_test) # Save testing labels

np.savez(ARTIFACTS_DIR / "X_train.npz", X=X_train_processed) # Save processed training features
np.savez(ARTIFACTS_DIR / "X_test.npz", X=X_test_processed) # Save processed testing features

print(f"\nArtifacts saved to: {ARTIFACTS_DIR.resolve()}") # Print path
