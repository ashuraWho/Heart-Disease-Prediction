# ============================================================
# MODULE 02 – CLASSICAL MACHINE LEARNING
# Heart Disease Prediction
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

from joblib import load, dump

# =====================
# PATHS
# =====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# =====================
# LOAD ARTIFACTS
# =====================

X_train = np.load(ARTIFACTS_DIR / "X_train.npz")["X"]
X_test  = np.load(ARTIFACTS_DIR / "X_test.npz")["X"]
y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
y_test  = np.load(ARTIFACTS_DIR / "y_test.npy")

print("Artifacts loaded successfully")

# =====================
# EVALUATION FUNCTION
# =====================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

# =====================
# BASELINE
# =====================

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_results = evaluate_model(baseline, X_test, y_test)

# =====================
# CV STRATEGY
# =====================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================
# MODELS
# =====================

models = {
    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        ),
        {"C": [0.01, 0.1, 1, 10]}
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9, 11]}
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced"),
        {"C": [0.1, 1, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {"n_estimators": [100, 300], "max_depth": [None, 5, 10]}
    )
}

results = {"Baseline": baseline_results}
best_models = {}

# =====================
# TRAINING LOOP
# =====================

for name, (model, params) in models.items():
    gs = GridSearchCV(
        model,
        params,
        cv=cv,
        scoring="recall",
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    best_models[name] = gs.best_estimator_
    results[name] = evaluate_model(gs.best_estimator_, X_test, y_test)

# =====================
# RESULTS SUMMARY
# =====================

results_df = pd.DataFrame(results).T
print("\nFinal Model Comparison:")
print(results_df)

# =====================
# ROC CURVES
# =====================

plt.figure(figsize=(8, 6))
for name, model in best_models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

plt.title("ROC Curve Comparison")
plt.show()

# =====================
# BEST MODEL
# =====================

best_model_name = results_df["Recall"].idxmax()
best_model = best_models[best_model_name]

print(f"\nBest model by Recall: {best_model_name}")

# =====================
# CONFUSION MATRIX
# =====================

cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix – {best_model_name}")
plt.show()

# =====================
# SAVE BEST MODEL
# =====================

dump(best_model, ARTIFACTS_DIR / "best_model_classic.joblib")