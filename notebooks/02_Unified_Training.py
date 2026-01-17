# ============================================================ #
# Module 02 – Unified Model Training & Selection               #
# ============================================================ #

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from shared_utils import (
        setup_environment, 
        console,
        ARTIFACTS_DIR
    )
except ImportError:
    print("Error: shared_utils not found.")
    sys.exit(1)

# Initialize Environment
setup_environment()

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# --- ML IMPORTS ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# --- DL IMPORTS ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# --- REPRODUCIBILITY ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- LOAD DATA ---
console.print("[bold cyan]Loading preprocessed data...[/bold cyan]")
try:
    X_train = np.load(ARTIFACTS_DIR / "X_train.npz")["X"]
    X_test  = np.load(ARTIFACTS_DIR / "X_test.npz")["X"]
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
    y_test  = np.load(ARTIFACTS_DIR / "y_test.npy")
except Exception as e:
    console.print(f"[bold red]ERROR: Could not load artifacts: {e}[/bold red]")
    console.print("[yellow]>>> Please run Module 01 first.[/yellow]")
    sys.exit(1)

input_dim = X_train.shape[1]

# --- EVALUATION HELPER ---
def evaluate(name, y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    
    console.print(f"\n[bold underline]{name} Results[/bold underline]")
    console.print(f"Accuracy:  {acc:.4f}")
    console.print(f"Precision: {prec:.4f}")
    console.print(f"Recall:    {rec:.4f}")
    console.print(f"F1-Score:  {f1:.4f}")
    console.print(f"ROC-AUC:   {auc:.4f}")
    return f1

# --- 1. CLASSICAL ML ---
console.print("\n[bold header][STEP 1] Tuning Classical Models...[/bold header]")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models_to_tune = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"), 
        {"C": [0.1, 1, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, class_weight="balanced"), 
        {"n_estimators": [100, 200], "max_depth": [5, 10]}
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced"), 
        {"C": [0.1, 1, 10]}
    )
}

best_sklearn_model = None
best_sklearn_f1 = -1
best_sklearn_name = ""

for name, (model, params) in models_to_tune.items():
    with console.status(f"Training {name}...", spinner="dots"):
        gs = GridSearchCV(model, params, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
    
    y_pred = gs.best_estimator_.predict(X_test)
    y_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
    
    current_f1 = evaluate(name, y_test, y_pred, y_proba)
    
    if current_f1 > best_sklearn_f1:
        best_sklearn_f1 = current_f1
        best_sklearn_model = gs.best_estimator_
        best_sklearn_name = name

console.print(f"\n[bold green]Best Classical Model: {best_sklearn_name} (F1: {best_sklearn_f1:.4f})[/bold green]")

# --- 2. DEEP LEARNING ---
console.print("\n[bold header][STEP 2] Training Neural Network (DL)...[/bold header]")

nn_model = Sequential([
    Input(shape=(input_dim,)),
    Dense(32, activation="relu", kernel_regularizer=l2(1e-3)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation="relu", kernel_regularizer=l2(1e-3)),
    Dense(1, activation="sigmoid")
])

nn_model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

with console.status("Training Neural Network...", spinner="dots"):
    history = nn_model.fit(
        X_train, y_train, 
        validation_split=0.2, 
        epochs=150, 
        batch_size=16, 
        callbacks=[early_stop], 
        verbose=0
    )

y_proba_nn = nn_model.predict(X_test, verbose=0).ravel()
y_pred_nn = (y_proba_nn >= 0.5).astype(int)
dl_f1 = evaluate("Deep Learning (MLP)", y_test, y_pred_nn, y_proba_nn)

# --- 3. SELECTION & SAVE ---
console.print("\n[bold header][STEP 3] Selecting Final Winner...[/bold header]")

if dl_f1 > best_sklearn_f1:
    console.print(f"[bold green]WINNER: Deep Learning (F1: {dl_f1:.4f})[/bold green]")
    nn_model.save(ARTIFACTS_DIR / "best_model_unified.keras")
    with open(ARTIFACTS_DIR / "model_type.txt", "w") as f: f.write("keras")
    final_pred = y_pred_nn
else:
    console.print(f"[bold green]WINNER: {best_sklearn_name} (F1: {best_sklearn_f1:.4f})[/bold green]")
    dump(best_sklearn_model, ARTIFACTS_DIR / "best_model_unified.joblib")
    with open(ARTIFACTS_DIR / "model_type.txt", "w") as f: f.write("sklearn")
    final_pred = best_sklearn_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Pipeline Winner")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

console.print("[bold cyan]Training Complete. Best model saved.[/bold cyan]")
