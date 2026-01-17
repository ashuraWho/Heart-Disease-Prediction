# ============================================================ #
# Module 01 – EDA & Preprocessing (Heart 2022)                 #
# ============================================================ #

import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from shared_utils import (
        setup_environment, 
        console, 
        DATASET_PATH, 
        ARTIFACTS_DIR
    )
except ImportError:
    print("Error: shared_utils not found.")
    sys.exit(1)

setup_environment()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

console.print(f"Initial Rows: [bold]{len(df)}[/bold]")

# --- 1. TARGET CREATION ---
# Target is "Yes" if HadHeartAttack or HadAngina is "Yes"
console.print("[cyan]Creating Target Variable (HeartDisease)...[/cyan]")
df['HeartDisease'] = 0
mask_disease = (df['HadHeartAttack'] == 'Yes') | (df['HadAngina'] == 'Yes')
df.loc[mask_disease, 'HeartDisease'] = 1

console.print(f"Target Distribution:\n{df['HeartDisease'].value_counts(normalize=True)}")

# --- 2. SIMPLE FEATURE ENGINEERING ---
# Health Score (GeneralHealth + PhysicalHealthDays)
# Map GeneralHealth to 1-5
gen_health_map = {"Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5}
df['GeneralHealth_Num'] = df['GeneralHealth'].map(gen_health_map).fillna(3) # Fill NaN with neutral

# Interaction: Sleep * Physical (Rest & Activity balance)
if 'SleepHours' in df.columns and 'PhysicalHealthDays' in df.columns:
    df['Sleep_Health_Ratio'] = df['SleepHours'] / (df['PhysicalHealthDays'] + 1)

# BMI Category is already calculated in dataset? 'BMI' column exists. Let's keep raw BMI.

# --- 3. DEFINE FEATURES FOR TRAINING ---
# We will drop the original target columns and irrelevant IDs (State)
drop_cols = ['State', 'HadHeartAttack', 'HadAngina', 'HeartDisease', 'LastCheckupTime', 'RemovedTeeth', 'TetanusLast10Tdap', 'FluVaxLast12', 'PneumoVaxEver', 'HIVTesting', 'HighRiskLastYear', 'CovidPos'] 
# Dropping some administrative/less relevant columns to keep dimensionality sane, 
# can add back later if needed. 'LastCheckupTime' is messy text.

# Filter only columns that exist
available_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=available_cols) 
y = df['HeartDisease']

console.print(f"Feature Columns ({X.shape[1]}): {X.columns.tolist()}")

# --- 4. PREPROCESSING PIPELINES ---
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Define specific Ordinal columns if any
ordinal_cols = ['GeneralHealth', 'AgeCategory', 'SmokerStatus'] # These have order
nominal_cols = [c for c in categorical_cols if c not in ordinal_cols]

# Mappings for Ordinal Encoder (Ensure strings match data exactly)
# Note: Sklearn OrdinalEncoder is tricky with explicit mappings. 
# For simplicity/robustness, we'll treat 'GeneralHealth' and 'Age' as Ordinal manually or let OneHot handle it safely if not explicitly mapped.
# Given cardinality of AgeCategory (13 levels) and SmokerStatus (4), order matters.
# Let's use OneHot for everything to be safe and avoid "Unknown category" errors, 
# unless we map them manually beforehand.
# Actually, manual mapping is safer for inference consistency.

# Manual Map for robust inference:
age_order = sorted(df['AgeCategory'].unique()) # A-Z sorts it correctly mostly? "Age 18-24", "Age 80 or older" -> Yes, alphanumeric sort roughly works except "80 or older" usually last.
# Let's stick to OneHot for maximum compatibility unless we see perf issues.
# Wait, Age "80 or older" sorts after "75-79". "Age 18-24" sorts first. 
# String sorting works for this specific format!
# We will use OneHotEncoder for ALL categories to make the pipeline 100% robust.

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ]
)

# --- 5. SPLIT & TRANSFORM ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

console.print("[cyan]Fitting Preprocessor...[/cyan]")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

console.print(f"Processed Shape: {X_train_processed.shape}")

# --- SAVE ARTIFACTS ---
dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
np.save(ARTIFACTS_DIR / "y_train.npy", y_train)
np.save(ARTIFACTS_DIR / "y_test.npy", y_test)
np.savez(ARTIFACTS_DIR / "X_train.npz", X=X_train_processed)
np.savez(ARTIFACTS_DIR / "X_test.npz", X=X_test_processed)
# Save columns used for inference knowing what to ask
dump(X.columns.tolist(), ARTIFACTS_DIR / "feature_names.joblib")

console.print(f"[bold green]Artifacts saved to {ARTIFACTS_DIR}[/bold green]")
