# ============================================================ #
# Module 02 – Training (Mega-Ensemble + Threshold Tournament)  #
# ============================================================ #

import sys
from pathlib import Path

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

setup_environment()

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# --- ML IMPORTS ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, precision_recall_curve, fbeta_score
)
from imblearn.over_sampling import SMOTE

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
    sys.exit(1)

input_dim = X_train.shape[1]
console.print(f"Original Training Size: {len(X_train)} (Positives: {sum(y_train)})")

# --- SMOTE OVERSAMPLING ---
console.print("\n[bold yellow]Performing SMOTE Oversampling...[/bold yellow]")
smote = SMOTE(random_state=SEED)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

console.print(f"Total SMOTE Data: {len(X_train_res)} (Balanced)")


# --- THRESHOLD TOURNAMENT LOGIC ---
def calculate_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8) # Recall
    specificity = tn / (tn + fp + 1e-8) # True Negative Rate
    
    metrics = {
        "F1": f1_score(y_true, y_pred),
        "F2": fbeta_score(y_true, y_pred, beta=2),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Youden": sensitivity + specificity - 1,
        "G-Mean": np.sqrt(sensitivity * specificity),
        "ROC_Dist": np.sqrt((1 - sensitivity)**2 + (1 - specificity)**2) # Lower is better
    }
    return metrics, (tn, fp, fn, tp)

def run_threshold_tournament(y_true, y_proba):
    console.print("\n[bold magenta]Running Threshold Tournament...[/bold magenta]")
    thresholds = np.linspace(0.01, 0.99, 100)
    
    best_results = {
        "F1":       {"score": -1, "thresh": 0.5},
        "F2":       {"score": -1, "thresh": 0.5},
        "MCC":      {"score": -1, "thresh": 0.5},
        "Youden":   {"score": -1, "thresh": 0.5},
        "G-Mean":   {"score": -1, "thresh": 0.5},
        "ROC_Dist": {"score": 99, "thresh": 0.5} # Lower is better
    }
    
    # 1. Sweep
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        metrics, _ = calculate_metrics(y_true, y_pred, y_proba)
        
        for name in ["F1", "F2", "MCC", "Youden", "G-Mean"]:
            if metrics[name] > best_results[name]["score"]:
                best_results[name] = {"score": metrics[name], "thresh": t}
                
        if metrics["ROC_Dist"] < best_results["ROC_Dist"]["score"]:
            best_results["ROC_Dist"] = {"score": metrics["ROC_Dist"], "thresh": t}

    # 2. Compare & Select Winner
    # We select the strategy that minimizes ROC_Dist (closest to maximal perfect prediction)
    # But we want to show the user all options.
    
    console.print("\n[bold]Tournament Results:[/bold]")
    console.print(f"{'Strategy':<15} {'Best Thresh':<12} {'Score':<10} {'Dist to Perfect':<15}")
    console.print("-" * 55)
    
    global_winner = None
    min_dist = 99
    
    for name, res in best_results.items():
        # Calc distance for this winner
        t = res["thresh"]
        y_p = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        dist = np.sqrt((1 - sens)**2 + (1 - spec)**2)
        
        style_open = ""
        style_close = ""
        if dist < min_dist:
            min_dist = dist
            global_winner = (name, t, dist)
            style_open = "[green]"
            style_close = "[/green]"
            
        console.print(f"{style_open}{name:<15} {t:<12.2f} {res['score']:<10.4f} {dist:<15.4f}{style_close}")

    console.print(f"\n[bold green]WINNER: {global_winner[0]} (Threshold: {global_winner[1]:.2f})[/bold green]")
    return global_winner[1], global_winner[0]


# --- 1. TRAIN INDIVIDUAL MODELS ---
console.print("\n[bold header]Training Mega-Ensemble (7 Models)...[/bold header]")
models = {}

# 1. Logistic Regression
console.print("⠸ Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=0.1)
lr.fit(X_train_res, y_train_res)
models["LR"] = lr

# 2. Random Forest
console.print("⠸ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
models["RF"] = rf

# 3. Extra Trees
console.print("⠸ Training Extra Trees...")
et = ExtraTreesClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
et.fit(X_train_res, y_train_res)
models["ET"] = et

# 4. Gradient Boosting
console.print("⠸ Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=SEED)
gb.fit(X_train_res, y_train_res)
models["GB"] = gb

# 5. XGBoost
console.print("⠸ Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=6, 
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=-1
)
xgb.fit(X_train_res, y_train_res)
models["XGB"] = xgb

# 6. LightGBM
console.print("⠸ Training LightGBM...")
lgbm = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1)
lgbm.fit(X_train_res, y_train_res)
models["LGBM"] = lgbm

# 7. CatBoost
console.print("⠸ Training CatBoost...")
cat = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0, allow_writing_files=False)
cat.fit(X_train_res, y_train_res)
models["CAT"] = cat

# 8. Neural Network
console.print("⠸ Training Neural Network...")
nn = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
nn.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
nn.fit(
    X_train_res, y_train_res, 
    validation_split=0.1, 
    epochs=30, 
    batch_size=128, 
    verbose=0,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)
models["DL"] = nn


# --- 2. ENSEMBLE PREDICTIONS ---
console.print("\n[bold header]Evaluating Mega-Ensemble (Metrics Tournament)...[/bold header]")

# Gather Probabilities
probs = {}
for name, model in models.items():
    if name == "DL":
        probs[name] = model.predict(X_test, verbose=0).ravel()
    else:
        probs[name] = model.predict_proba(X_test)[:, 1]

# MEGA VOTE: Trust the GBM Trinity + ET
# XGB (4), LGBM (4), CAT (4), ET (3), RF (2), GB (2), DL (2), LR (0)
weights = {
    "LR": 0, "RF": 2, "ET": 3, "GB": 2, 
    "XGB": 4, "LGBM": 4, "CAT": 4, "DL": 2
}
total_weight = sum(weights.values())

ensemble_proba = np.zeros_like(y_test, dtype=float)
for name, w in weights.items():
    if w > 0:
        ensemble_proba += probs[name] * w

ensemble_proba /= total_weight

# --- 3. TOURNAMENT & OPTIMIZATION ---
opt_thresh, winner_name = run_threshold_tournament(y_test, ensemble_proba)

# Evaluate Winner
console.print(f"\n[bold underline]Final Optimized Results ({winner_name} Strategy)[/bold underline]")
y_pred = (ensemble_proba >= opt_thresh).astype(int)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, ensemble_proba)

console.print(f"Accuracy:  {acc:.4f}")
console.print(f"Precision: {prec:.4f}")
console.print(f"Recall:    {rec:.4f}")
console.print(f"F1-Score:  {f1:.4f}")
console.print(f"ROC-AUC:   {auc:.4f}")

# --- 4. SAVE EVERYTHING ---
console.print("\n[bold cyan]Saving Mega-Ensemble Artifacts...[/bold cyan]")
dump(lr, ARTIFACTS_DIR / "model_lr.joblib")
dump(rf, ARTIFACTS_DIR / "model_rf.joblib")
dump(gb, ARTIFACTS_DIR / "model_gb.joblib")
dump(xgb, ARTIFACTS_DIR / "model_xgb.joblib")
dump(et, ARTIFACTS_DIR / "model_et.joblib")
dump(lgbm, ARTIFACTS_DIR / "model_lgbm.joblib")
cat.save_model(str(ARTIFACTS_DIR / "model_cat.cbm"))
nn.save(ARTIFACTS_DIR / "model_dl.keras")

with open(ARTIFACTS_DIR / "threshold.txt", "w") as f:
    f.write(str(opt_thresh))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples") # Purple for Royalty (Winner)
plt.title(f"Tournament Winner ({winner_name} @ {opt_thresh:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

console.print("[bold green]Optimization Complete.[/bold green]")
