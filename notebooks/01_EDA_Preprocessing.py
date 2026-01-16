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

sns.set(style="whitegrid") # Redundant set style (maintained for completeness)
plt.rcParams["figure.figsize"] = (10, 6) # Redundant set rcParams (maintained for completeness)

RANDOM_STATE = 42 # Set a random state for reproducibility
TEST_SIZE = 0.2 # Set the proportion of the dataset to include in the test split

# ===================== # Header Section
# 3. PATH HANDLING      # Path Handling Header
# ===================== # Header Section

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Define the root directory of the project
DATA_DIR = PROJECT_ROOT / "data" # Define the data directory path
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

DATASET_PATH = DATA_DIR / "Heart_Disease_Prediction.csv" # Define the path to the dataset CSV file

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) # Create the artifacts directory if it doesn't exist

# ===================== # Header Section
# 4. LOAD DATASET       # Load Dataset Header
# ===================== # Header Section

df = pd.read_csv(DATASET_PATH) # Load the dataset into a pandas DataFrame

print(f"Dataset loaded from: {DATASET_PATH}") # Print the dataset source path
print(df.head()) # Print the first five rows of the DataFrame

# ===================== # Header Section
# 5. DATASET OVERVIEW    # Dataset Overview Header
# ===================== # Header Section

print(f"Number of rows: {df.shape[0]}") # Print the total number of rows in the dataset
print(f"Number of columns: {df.shape[1]}") # Print the total number of columns in the dataset
df.info() # Print a concise summary of the DataFrame

# ===================== # Header Section
# 6. COLUMN RENAMING & TARGET ENCODING # Renaming and Encoding Header
# ===================== # Header Section

rename_dict = { # Define a dictionary for renaming columns to shorter, clearer names
    "Chest pain type": "ChestPain", # Mapping 'Chest pain type' to 'ChestPain'
    "FBS over 120": "FBS", # Mapping 'FBS over 120' to 'FBS'
    "EKG results": "EKG", # Mapping 'EKG results' to 'EKG'
    "Max HR": "MaxHR", # Mapping 'Max HR' to 'MaxHR'
    "Exercise angina": "ExerciseAngina", # Mapping 'Exercise angina' to 'ExerciseAngina'
    "ST depression": "ST_Depression", # Mapping 'ST depression' to 'ST_Depression'
    "Slope of ST": "ST_Slope", # Mapping 'Slope of ST' to 'ST_Slope'
    "Number of vessels fluro": "NumVessels", # Mapping 'Number of vessels fluro' to 'NumVessels'
    "Heart Disease": "HeartDisease", # Mapping 'Heart Disease' to 'HeartDisease'
} # End of dictionary definition

df.rename(columns=rename_dict, inplace=True) # Apply the renaming to the DataFrame in place

df["HeartDisease"] = df["HeartDisease"].map({ # Map target labels to numerical values
    "Absence": 0, # Map 'Absence' of heart disease to 0
    "Presence": 1 # Map 'Presence' of heart disease to 1
}) # End of mapping

print("Columns after renaming:") # Print message indicating upcoming column list
print(df.columns) # Print the updated column names

# ===================== # Header Section
# 7. DATA QUALITY CHECKS # Data Quality Checks Header
# ===================== # Header Section

print("\nMissing values per column:") # Print header for missing values report
print(df.isnull().sum()) # Print the count of missing values for each column

duplicates = df.duplicated().sum() # Count the number of duplicate rows in the DataFrame
print(f"\nNumber of duplicate rows: {duplicates}") # Print the count of duplicate rows

# ===================== # Header Section
# 8. TARGET ANALYSIS    # Target Analysis Header
# ===================== # Header Section

sns.countplot(x="HeartDisease", data=df) # Create a count plot for the target variable
plt.title("Target Distribution (HeartDisease)") # Set the title of the target distribution plot
plt.show() # Display the target distribution plot

print(df["HeartDisease"].value_counts(normalize=True) * 100) # Print the normalized value counts (percentages) for the target

# ===================== # Header Section
# 9. FEATURE GROUPS     # Feature Groups Header
# ===================== # Header Section

numerical_features = [ # Define a list of numerical features
    "Age", "BP", "Cholesterol", "MaxHR", "ST_Depression" # Features: Age, Blood Pressure, Cholesterol, Max HR, ST Depression
] # End of numerical features list

categorical_features = [ # Define a list of categorical features
    "Sex", "ChestPain", "FBS", "EKG", # Features: Sex, Chest Pain, FBS, EKG
    "ExerciseAngina", "ST_Slope", "NumVessels", "Thallium" # Features: Exercise Angina, ST Slope, Num Vessels, Thallium
] # End of categorical features list

# ===================== # Header Section
# 10. EDA – NUMERICAL FEATURES # Numerical EDA Header
# ===================== # Header Section

for col in numerical_features: # Iterate through each numerical feature for visualization
    sns.histplot(df[col], kde=True) # Create a histogram with a Kernel Density Estimate (KDE)
    plt.title(f"Distribution of {col}") # Set the title for the feature distribution plot
    plt.show() # Display the distribution plot

    sns.boxplot(x="HeartDisease", y=col, data=df) # Create a box plot of the feature versus the target
    plt.title(f"{col} vs HeartDisease") # Set the title for the box plot
    plt.show() # Display the box plot

# ===================== # Header Section
# 11. EDA – CATEGORICAL FEATURES # Categorical EDA Header
# ===================== # Header Section

for col in categorical_features: # Iterate through each categorical feature for visualization
    sns.countplot(x=col, hue="HeartDisease", data=df) # Create a count plot of the feature with target hue
    plt.title(f"{col} vs HeartDisease") # Set the title for the categorical relationship plot
    plt.legend(title="HeartDisease") # Add a legend for the target variable
    plt.show() # Display the count plot

# ===================== # Header Section
# 12. CORRELATION ANALYSIS # Correlation Analysis Header
# ===================== # Header Section

corr_matrix = df[numerical_features + ["HeartDisease"]].corr() # Calculate the correlation matrix for numerical features and target
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm") # Create a heatmap of the correlation matrix with annotations
plt.title("Correlation Matrix") # Set the title for the correlation heatmap
plt.show() # Display the correlation heatmap

# ===================== # Header Section
# 13. FEATURE / TARGET SPLIT # Data Split Header
# ===================== # Header Section

X = df.drop("HeartDisease", axis=1) # Create the feature matrix by dropping the target column
y = df["HeartDisease"] # Create the target vector

# ===================== # Header Section
# 14. TRAIN / TEST SPLIT # Train/Test Split Header
# ===================== # Header Section

X_train, X_test, y_train, y_test = train_test_split( # Split data into training and testing sets
    X, # Feature matrix
    y, # Target vector
    test_size=TEST_SIZE, # Proportion of test set
    random_state=RANDOM_STATE, # Random state for reproducibility
    stratify=y # Stratify split based on target to maintain class proportions
) # End of split function

print("Train shape:", X_train.shape) # Print the shape of the training feature set
print("Test shape:", X_test.shape) # Print the shape of the testing feature set

# ===================== # Header Section
# 15. PREPROCESSING PIPELINE # Preprocessing Pipeline Header
# ===================== # Header Section

numeric_transformer = Pipeline(steps=[ # Define a pipeline for numerical feature transformation
    ("scaler", StandardScaler()) # Step 1: Scale features using StandardScaler
]) # End of numeric pipeline

categorical_transformer = Pipeline(steps=[ # Define a pipeline for categorical feature transformation
    ("onehot", OneHotEncoder( # Step 1: Encode features using OneHotEncoder
        drop="first", # Drop the first category to avoid multicollinearity
        handle_unknown="ignore" # Ignore unknown categories during transformation
    )) # End of OneHotEncoder config
]) # End of categorical pipeline

preprocessor = ColumnTransformer( # Combine transformations using ColumnTransformer
    transformers=[ # List of transformers
        ("num", numeric_transformer, numerical_features), # Apply numeric pipeline to numerical features
        ("cat", categorical_transformer, categorical_features) # Apply categorical pipeline to categorical features
    ] # End of transformer list
) # End of ColumnTransformer definition

# ===================== # Header Section
# 16. FIT & TRANSFORM   # Fit and Transform Header
# ===================== # Header Section

X_train_processed = preprocessor.fit_transform(X_train) # Fit preprocessor to training data and transform it
X_test_processed = preprocessor.transform(X_test) # Transform testing data using the fitted preprocessor

print("Shape after preprocessing (train):", X_train_processed.shape) # Print training shape after preprocessing
print("Shape after preprocessing (test):", X_test_processed.shape) # Print testing shape after preprocessing

# ===================== # Header Section
# 17. SAVE ARTIFACTS    # Save Artifacts Header
# ===================== # Header Section

dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib") # Save the fitted preprocessor to a joblib file

np.save(ARTIFACTS_DIR / "y_train.npy", y_train) # Save training labels to a numpy file
np.save(ARTIFACTS_DIR / "y_test.npy", y_test) # Save testing labels to a numpy file

np.savez(ARTIFACTS_DIR / "X_train.npz", X=X_train_processed) # Save processed training features to a compressed numpy archive
np.savez(ARTIFACTS_DIR / "X_test.npz", X=X_test_processed) # Save processed testing features to a compressed numpy archive

print(f"\nArtifacts saved to: {ARTIFACTS_DIR.resolve()}") # Print the absolute path where artifacts were saved
