# ============================================================
# NOTEBOOK 1 – EDA & PREPROCESSING
# Heart Disease Prediction Dataset
# ============================================================

# =====================
# 1. IMPORT LIBRARIES
# =====================

from pathlib import Path

import pandas as pd                  # Data manipulation and tabular data handling
import numpy as np                   # Numerical computations and array operations

import matplotlib.pyplot as plt      # Low-level plotting library
import seaborn as sns                # Statistical data visualization built on matplotlib

sns.set(style="whitegrid")           # Set a clean, publication-ready plotting style
plt.rcParams["figure.figsize"] = (10, 6)  # Default figure size for consistency across plots

from sklearn.model_selection import train_test_split   # Train/test splitting utilities
from sklearn.preprocessing import StandardScaler       # Feature scaling for numerical variables
from sklearn.preprocessing import OneHotEncoder        # Encoding for categorical variables
from sklearn.compose import ColumnTransformer           # Apply different preprocessing to different columns
from sklearn.pipeline import Pipeline                   # Build reproducible ML pipelines

from joblib import dump

# =====================
# 2. GLOBAL CONFIG
# =====================

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# =====================
# 3. PATH HANDLING (PORTABLE)
# =====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASET_PATH = DATA_DIR / "Heart_Disease_Prediction.csv"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# 4. LOAD DATASET
# =====================

df = pd.read_csv(DATASET_PATH)

print(f"Dataset loaded from: {DATASET_PATH}")
print(df.head())

# =====================
# 5. DATASET OVERVIEW
# =====================

print(f"Numero di righe: {df.shape[0]}")
print(f"Numero di colonne: {df.shape[1]}")
df.info()

# =====================
# 6. COLUMN RENAMING & TARGET ENCODING
# =====================

rename_dict = {
    "Chest pain type": "ChestPain",
    "FBS over 120": "FBS",
    "EKG results": "EKG",
    "Max HR": "MaxHR",
    "Exercise angina": "ExerciseAngina",
    "ST depression": "ST_Depression",
    "Slope of ST": "ST_Slope",
    "Number of vessels fluro": "NumVessels",
    "Heart Disease": "HeartDisease",
}

df.rename(columns=rename_dict, inplace=True)

df["HeartDisease"] = df["HeartDisease"].map({
    "Absence": 0,
    "Presence": 1
})

print("Columns after renaming:")
print(df.columns)

# =====================
# 7. DATA QUALITY CHECKS
# =====================

print("\nMissing values per column:")
print(df.isnull().sum())

duplicates = df.duplicated().sum()
print(f"\nNumero di righe duplicate: {duplicates}")

# =====================
# 8. TARGET ANALYSIS
# =====================

sns.countplot(x="HeartDisease", data=df)
plt.title("Distribuzione del Target (HeartDisease)")
plt.show()

print(df["HeartDisease"].value_counts(normalize=True) * 100)

# =====================
# 9. FEATURE GROUPS
# =====================

numerical_features = [
    "Age", "BP", "Cholesterol", "MaxHR", "ST_Depression"
]

categorical_features = [
    "Sex", "ChestPain", "FBS", "EKG",
    "ExerciseAngina", "ST_Slope", "NumVessels", "Thallium"
]

# =====================
# 10. EDA – NUMERICAL FEATURES
# =====================

for col in numerical_features:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribuzione di {col}")
    plt.show()

    sns.boxplot(x="HeartDisease", y=col, data=df)
    plt.title(f"{col} vs HeartDisease")
    plt.show()

# =====================
# 11. EDA – CATEGORICAL FEATURES
# =====================

for col in categorical_features:
    sns.countplot(x=col, hue="HeartDisease", data=df)
    plt.title(f"{col} vs HeartDisease")
    plt.legend(title="HeartDisease")
    plt.show()

# =====================
# 12. CORRELATION ANALYSIS
# =====================

corr_matrix = df[numerical_features + ["HeartDisease"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# =====================
# 13. FEATURE / TARGET SPLIT
# =====================

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =====================
# 14. TRAIN / TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =====================
# 15. PREPROCESSING PIPELINE
# =====================

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =====================
# 16. FIT & TRANSFORM
# =====================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Shape after preprocessing (train):", X_train_processed.shape)
print("Shape after preprocessing (test):", X_test_processed.shape)

# =====================
# 17. SAVE ARTIFACTS
# =====================

dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

np.save(ARTIFACTS_DIR / "y_train.npy", y_train)
np.save(ARTIFACTS_DIR / "y_test.npy", y_test)

np.savez(ARTIFACTS_DIR / "X_train.npz", X=X_train_processed)
np.savez(ARTIFACTS_DIR / "X_test.npz", X=X_test_processed)

print(f"\nArtifacts saved to: {ARTIFACTS_DIR.resolve()}")