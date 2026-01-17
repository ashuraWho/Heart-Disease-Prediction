# ============================================================ #
# Module 01 – EDA & Preprocessing                              #
# ============================================================ #

import sys
from pathlib import Path

# Add current directory to sys.path to ensure shared_utils can be imported
# if run directly or via main.py
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from shared_utils import (
        setup_environment, 
        console, 
        DATASET_PATH, 
        ARTIFACTS_DIR, 
        CLINICAL_GUIDE
    )
except ImportError:
    # Fallback if shared_utils is not found (shouldn't happen with correct sys.path)
    print("Error: shared_utils not found.")
    sys.exit(1)

# Initialize Environment
setup_environment()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump

# --- GLOBAL CONFIG ---
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- LOAD DATASET ---
console.print(f"[bold cyan]Loading dataset from:[/bold cyan] {DATASET_PATH}")

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    console.print(f"[bold red]ERROR: Dataset not found at {DATASET_PATH}[/bold red]")
    sys.exit(1)

# --- DATA CLEANING ---
initial_rows = df.shape[0]
df.dropna(inplace=True)
final_rows = df.shape[0]
console.print(f"Rows: [bold]{initial_rows}[/bold] -> [bold green]{final_rows}[/bold] (Dropped {initial_rows - final_rows} NaNs)")

# --- MAPPING FUNCTIONS ---
# We map strings to numbers to enable correlation analysis and visualization.

# 1. Binary Mapping
binary_cols = ["Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure", 
               "Low HDL Cholesterol", "High LDL Cholesterol"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

if "Heart Disease Status" in df.columns:
    df["HeartDisease"] = df["Heart Disease Status"].map({"Yes": 1, "No": 0})
    df.drop(columns=["Heart Disease Status"], inplace=True)

# 2. Ordinal Level Mapping
level_mapping = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
level_mapping_basic = {"Low": 0, "Medium": 1, "High": 2}

ordinal_map_cols = {
    "Exercise Habits": level_mapping_basic,
    "Stress Level": level_mapping_basic,
    "Sugar Consumption": level_mapping_basic,
    "Alcohol Consumption": level_mapping
}

for col, mapping in ordinal_map_cols.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# --- INTERACTIVE EDA FUNCTIONS ---
def show_correlation_matrix():
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numerical Correlation Matrix")
    plt.tight_layout()
    plt.show()

def show_target_distribution():
    plt.figure(figsize=(6, 4))
    sns.countplot(x="HeartDisease", data=df)
    plt.title("Distribution of Heart Disease Presence")
    plt.show()
    
def show_feature_plots():
    for col in df.drop("HeartDisease", axis=1).columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        if df[col].nunique() > 5: # Assume continuous if specific number of unique vals
             sns.histplot(df[col], kde=True)
        else:
             sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x="HeartDisease", y=col, data=df)
        plt.title(f"{col} vs Heart Disease")
        
        plt.tight_layout()
        plt.show()
        
        cont = input("Press Enter for next, 'q' to stop: ")
        if cont.lower() == 'q': break

# --- PREPROCESSING & SPLIT ---
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

numerical_cols = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numerical_cols)]
)

# Fit & Transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- SAVE ARTIFACTS ---
dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
np.save(ARTIFACTS_DIR / "y_train.npy", y_train)
np.save(ARTIFACTS_DIR / "y_test.npy", y_test)
np.savez(ARTIFACTS_DIR / "X_train.npz", X=X_train_processed)
np.savez(ARTIFACTS_DIR / "X_test.npz", X=X_test_processed)

console.print(f"[bold green]Artifacts saved to {ARTIFACTS_DIR}[/bold green]")

# --- ENTRY POINT ---
if __name__ == "__main__":
    if "--plots" in sys.argv:
        # Simple menu for plotting if run directly with --plots
        while True:
            console.print("\n[bold]EDA Menu:[/bold] [1] Correlation [2] Target Dist [3] Features [q] Quit")
            choice = input("Select: ").strip().lower()
            if choice == '1': show_correlation_matrix()
            elif choice == '2': show_target_distribution()
            elif choice == '3': show_feature_plots()
            elif choice == 'q': break
