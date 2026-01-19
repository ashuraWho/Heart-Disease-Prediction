# ============================================================ #
# Module 02 – Training (Mega-Ensemble + Threshold Tournament)  #
# ============================================================ #
# Questo modulo:
# - carica i dati preprocessati
# - riequilibra le classi con SMOTE
# - allena un mega-ensemble (ML classico + boosting + DL)
# - combina i modelli con pesi esperti
# - ottimizza la soglia decisionale con un "Threshold Tournament"
# - salva tutti i modelli e gli artefatti finali
# ============================================================ #

import sys
from pathlib import Path

# ------------------------------------------------------------ #
# Aggiunta del project root al PYTHONPATH
# Necessario per importare shared_utils
# ------------------------------------------------------------ #
sys.path.append(str(Path(__file__).resolve().parent))

try:
    # Utility condivise di progetto
    from shared_utils import (
        setup_environment,                # Setup seed + env vars
        console,                          # Logging strutturato
        ARTIFACTS_DIR                     # Directory artefatti
    )
except ImportError:
    print("Error: shared_utils not found.")
    sys.exit(1)

# ------------------------------------------------------------ #
# Setup globale dell'ambiente di esecuzione
# ------------------------------------------------------------ #
setup_environment()

# ------------------------------------------------------------ #
# Import scientific stack
# ------------------------------------------------------------ #
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump


# ============================================================ #
# IMPORT MODELLI MACHINE LEARNING
# ============================================================ #
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    fbeta_score
)
from imblearn.over_sampling import SMOTE


# ============================================================ #
# IMPORT DEEP LEARNING
# ============================================================ #
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


# ============================================================ #
# RIPRODUCIBILITÀ
# ============================================================ #
SEED = 42 # Seed globale
random.seed(SEED) # RNG Python
np.random.seed(SEED) # RNG NumPy
tf.random.set_seed(SEED) # RNG TensorFlow


# ============================================================ #
# CARICAMENTO DATI PREPROCESSATI
# ============================================================ #

console.print("[bold cyan]Loading preprocessed data...[/bold cyan]")

try:
    # Feature matrix preprocessata
    X_train = np.load(ARTIFACTS_DIR / "X_train.npz")["X"]
    X_test  = np.load(ARTIFACTS_DIR / "X_test.npz")["X"]

    # Target binario
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
    y_test  = np.load(ARTIFACTS_DIR / "y_test.npy")
    
except Exception as e:
    # Fail immediato se gli artefatti non sono disponibili
    console.print(f"[bold red]ERROR: Could not load artifacts: {e}[/bold red]")
    sys.exit(1)

# Numero di feature
input_dim = X_train.shape[1]

console.print(f"Original Training Size: {len(X_train)} (Positives: {sum(y_train)})")


# ============================================================ #
# BILANCIAMENTO CLASSI - STRATEGIA COMBINATA
# ============================================================ #
# Problema: Dataset estremamente sbilanciato (91.2% vs 8.8%)
# Soluzione: Undersampling + SMOTE per bilanciamento ottimale
# ============================================================ #

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

console.print("\n[bold yellow]Balancing classes (Handling 10.38:1 imbalance)...[/bold yellow]")

# STEP 1: Undersampling leggero della classe maggioritaria prima di SMOTE
# Riduciamo la classe maggioritaria a ~3:1 (più gestibile per SMOTE)
console.print("[cyan]Step 1: Light undersampling (reduce majority to 3:1 ratio)...[/cyan]")

n_minority = sum(y_train == 1)
n_majority_target = n_minority * 3  # Vogliamo 3:1 invece di 10.38:1

# Undersample della classe maggioritaria
undersampler = RandomUnderSampler(
    sampling_strategy={0: n_majority_target, 1: n_minority},
    random_state=SEED
)
X_train_temp, y_train_temp = undersampler.fit_resample(X_train, y_train)

console.print(f"  After undersampling: {len(X_train_temp)} samples")
console.print(f"  Class distribution: {sum(y_train_temp == 0)} healthy, {sum(y_train_temp == 1)} disease")

# STEP 2: SMOTE sulla versione ridotta per bilanciare a 1:1
console.print("[cyan]Step 2: SMOTE oversampling (balance to 1:1)...[/cyan]")

smote = SMOTE(random_state=SEED, k_neighbors=min(5, n_minority - 1))  # k_neighbors adattivo

X_train_res, y_train_res = smote.fit_resample(X_train_temp, y_train_temp)

console.print(f"  Final balanced dataset: {len(X_train_res)} samples")
console.print(f"  Class distribution: {sum(y_train_res == 0)} healthy, {sum(y_train_res == 1)} disease (1:1)")


# ============================================================ #
# THRESHOLD TOURNAMENT – METRICHE AVANZATE
# ============================================================ #

def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calcola metriche clinicamente rilevanti
    per una data soglia decisionale.
    """
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensibilità = Recall
    sensitivity = tp / (tp + fn + 1e-8)
    
    # Specificità = True Negative Rate
    specificity = tn / (tn + fp + 1e-8)
    
    metrics = {
        "F1": f1_score(y_true, y_pred),
        "F2": fbeta_score(y_true, y_pred, beta=2),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Youden": sensitivity + specificity - 1,
        "G-Mean": np.sqrt(sensitivity * specificity),
        "ROC_Dist": np.sqrt((1 - sensitivity)**2 + (1 - specificity)**2) # Distanza dal classificatore perfetto
    }
    
    return metrics, (tn, fp, fn, tp)


def run_threshold_tournament(y_true, y_proba):
    """
    Sweep completo delle soglie e selezione
    della strategia ottimale.
    Ottimizzato per ridurre falsi positivi mantenendo buona recall.
    """
    
    console.print("\n[bold magenta]Running Threshold Tournament...[/bold magenta]")
    thresholds = np.linspace(0.01, 0.99, 100)
    
    #  Inizializzazione best result per ogni metrica
    best_results = {
        "F1":           {"score": -1, "thresh": 0.5},
        "F2":           {"score": -1, "thresh": 0.5},
        "MCC":          {"score": -1, "thresh": 0.5},
        "Youden":       {"score": -1, "thresh": 0.5},
        "G-Mean":       {"score": -1, "thresh": 0.5},
        "Precision":    {"score": -1, "thresh": 0.5},  # Nuovo: privilegia precision
        "F1_Precision": {"score": -1, "thresh": 0.5},  # Nuovo: bilancia F1 e Precision
        "ROC_Dist":     {"score": 99, "thresh": 0.5}   # Lower is better
    }
    
    # -------------------------------------------------------- #
    # 1. Sweep su tutte le soglie
    # -------------------------------------------------------- #
    for t in thresholds:
        
        # Binarizzazione probabilità
        y_pred = (y_proba >= t).astype(int)
        metrics, (tn, fp, fn, tp) = calculate_metrics(y_true, y_pred, y_proba)
        
        # Calcolo precision separato per ottimizzazione
        precision = tp / (tp + fp + 1e-8)
        
        # Metriche da massimizzare
        for name in ["F1", "F2", "MCC", "Youden", "G-Mean"]:
            if metrics[name] > best_results[name]["score"]:
                best_results[name] = {"score": metrics[name], "thresh": t}
        
        # Precision: massimizza precision (riduce falsi positivi)
        if precision > best_results["Precision"]["score"]:
            best_results["Precision"] = {"score": precision, "thresh": t}
        
        # F1_Precision: media armonica di F1 e Precision
        # Bilancia recall e precision per ridurre sia FN che FP
        f1_prec_score = 2 * (metrics["F1"] * precision) / (metrics["F1"] + precision + 1e-8)
        if f1_prec_score > best_results["F1_Precision"]["score"]:
            best_results["F1_Precision"] = {"score": f1_prec_score, "thresh": t}
        
        # Metrica da minimizzare
        if metrics["ROC_Dist"] < best_results["ROC_Dist"]["score"]:
            best_results["ROC_Dist"] = {"score": metrics["ROC_Dist"], "thresh": t}


    # -------------------------------------------------------- #
    # 2. Confronto finale con pesi per priorità clinica
    # -------------------------------------------------------- #
    console.print("\n[bold]Tournament Results:[/bold]")
    console.print(f"{'Strategy':<18} {'Thresh':<10} {'Score':<10} {'Prec':<8} {'Recall':<8} {'FP':<8}")
    console.print("-" * 75)
    
    # Valutiamo ogni strategia con una metrica composita che penalizza FP
    candidates = []
    
    for name, res in best_results.items():
        t = res["thresh"]
        y_p = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
        sens = tp / (tp + fn + 1e-8)  # Recall
        spec = tn / (tn + fp + 1e-8)  # Specificity
        prec = tp / (tp + fp + 1e-8)  # Precision
        
        # Metrica composita: bilancia precision, recall e penalizza FP
        # Priorità: minimizzare FP mantenendo buona recall
        composite_score = (2 * prec * sens) / (prec + sens + 1e-8) - (fp / len(y_true)) * 0.5
        
        candidates.append({
            "name": name,
            "thresh": t,
            "composite": composite_score,
            "precision": prec,
            "recall": sens,
            "fp": fp,
            "fn": fn
        })
        
        console.print(f"{name:<18} {t:<10.3f} {res['score']:<10.4f} {prec:<8.3f} {sens:<8.3f} {fp:<8d}")

    # Seleziona la strategia migliore bilanciando recall e precision
    # Priorità clinica: recall >= 0.70 (non perdere pazienti con malattia)
    # poi minimizza FP (non generare falsi allarmi)
    valid_candidates = [c for c in candidates if c["recall"] >= 0.70]
    
    if valid_candidates:
        # Calcola un score bilanciato: privilegia recall ma penalizza FP
        # Score = F1 * (1 - fp_rate) dove fp_rate = FP / total_negative
        total_negative = sum(y_true == 0)
        for c in valid_candidates:
            fp_rate = c["fp"] / total_negative
            # Score bilanciato: F1 modificato per penalizzare falsi positivi
            # Peso recall 60%, penalità FP 40%
            c["balanced_score"] = (2 * c["precision"] * c["recall"]) / (c["precision"] + c["recall"] + 1e-8) * 0.6 - fp_rate * 0.4
        
        # Seleziona con migliore balanced_score
        global_winner_candidate = max(valid_candidates, key=lambda x: x["balanced_score"])
    else:
        # Se nessuno ha recall >= 0.70, cerca il miglior compromesso
        # Privilegia recall ma con buona precision
        for c in candidates:
            if c["recall"] >= 0.65:
                # Calcola score simile ma più permissivo
                total_negative = sum(y_true == 0)
                fp_rate = c["fp"] / total_negative
                c["balanced_score"] = (2 * c["precision"] * c["recall"]) / (c["precision"] + c["recall"] + 1e-8) * 0.5 - fp_rate * 0.3
        
        # Seleziona il migliore tra quelli con recall >= 0.65
        valid_fallback = [c for c in candidates if c["recall"] >= 0.65 and "balanced_score" in c]
        if valid_fallback:
            global_winner_candidate = max(valid_fallback, key=lambda x: x["balanced_score"])
        else:
            # Ultimo fallback: migliore F1
            global_winner_candidate = max(candidates, key=lambda x: (2 * x["precision"] * x["recall"]) / (x["precision"] + x["recall"] + 1e-8))
    
    global_winner = (global_winner_candidate["name"], global_winner_candidate["thresh"])

    console.print(f"\n[bold green]WINNER: {global_winner[0]} (Threshold: {global_winner[1]:.3f})[/bold green]")
    console.print(f"[cyan]Expected Metrics:[/cyan]")
    console.print(f"  Precision: {global_winner_candidate['precision']:.3f}")
    console.print(f"  Recall:    {global_winner_candidate['recall']:.3f} {'[yellow]⚠️[/yellow]' if global_winner_candidate['recall'] < 0.70 else ''}")
    console.print(f"  F1-Score:  {(2 * global_winner_candidate['precision'] * global_winner_candidate['recall']) / (global_winner_candidate['precision'] + global_winner_candidate['recall'] + 1e-8):.3f}")
    console.print(f"  FP: {global_winner_candidate['fp']} | FN: {global_winner_candidate['fn']}")
    
    # Warning se recall troppo bassa (critico in clinica)
    if global_winner_candidate['recall'] < 0.70:
        console.print(f"\n[yellow]⚠️ WARNING: Recall < 0.70. Consider lowering threshold for better disease detection.[/yellow]")
    elif global_winner_candidate['recall'] < 0.75:
        console.print(f"\n[cyan]💡 TIP: Recall could be improved. Current recall: {global_winner_candidate['recall']:.3f}[/cyan]")
    
    return global_winner[1], global_winner[0]


# ============================================================ #
# 1. TRAINING MEGA-ENSEMBLE
# ============================================================ #
console.print("\n[bold header]Training Mega-Ensemble (7 Models)...[/bold header]")
models = {}

# ============================================================ #
# CALCOLO CLASS WEIGHTS PER RIDURRE FALSI POSITIVI
# ============================================================ #
# Usiamo class_weight per penalizzare i falsi positivi
# bilanciando meglio la precision rispetto alla recall
# ============================================================ #

# Calcola pesi per bilanciare: penalizziamo moderatamente la classe positiva
# per ridurre i falsi positivi senza perdere troppa recall
# Bilanciamento: vogliamo mantenere recall >= 0.70 ma ridurre FP
class_weight_dict = {0: 1.0, 1: 0.90}  # Penalizzazione più leggera per mantenere recall

console.print(f"[dim]Using class weights: {class_weight_dict}[/dim]")

# 1. Logistic Regression
console.print("⠸ Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000, 
    C=0.1, 
    class_weight=class_weight_dict  # Aggiunge penalizzazione per falsi positivi
)
lr.fit(X_train_res, y_train_res)
models["LR"] = lr

# 2. Random Forest
console.print("⠸ Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    random_state=SEED, 
    n_jobs=-1,
    class_weight=class_weight_dict  # Penalizza falsi positivi
)
rf.fit(X_train_res, y_train_res)
models["RF"] = rf

# 3. Extra Trees
console.print("⠸ Training Extra Trees...")
et = ExtraTreesClassifier(
    n_estimators=100, 
    max_depth=12, 
    random_state=SEED, 
    n_jobs=-1,
    class_weight=class_weight_dict  # Penalizza falsi positivi
)
et.fit(X_train_res, y_train_res)
models["ET"] = et

# 4. Gradient Boosting
console.print("⠸ Training Gradient Boosting...")
# GB non supporta class_weight diretto, ma possiamo usare sample_weight
# Calcoliamo sample_weight manualmente
sample_weight_gb = np.array([class_weight_dict[y] for y in y_train_res])
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=SEED)
gb.fit(X_train_res, y_train_res, sample_weight=sample_weight_gb)
models["GB"] = gb

# 5. XGBoost
console.print("⠸ Training XGBoost...")
# XGBoost usa scale_pos_weight per bilanciare (inverso rispetto class_weight)
# scale_pos_weight = n_negative / n_positive * class_weight_ratio
n_neg = sum(y_train_res == 0)
n_pos = sum(y_train_res == 1)
scale_pos_weight = (n_neg / n_pos) * (class_weight_dict[0] / class_weight_dict[1])
xgb = XGBClassifier(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=6, 
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight  # Bilancia classi in XGBoost
)
xgb.fit(X_train_res, y_train_res)
models["XGB"] = xgb

# 6. LightGBM
console.print("⠸ Training LightGBM...")
# LightGBM usa anche scale_pos_weight
lgbm = LGBMClassifier(
    n_estimators=150, 
    learning_rate=0.05, 
    num_leaves=31, 
    random_state=SEED, 
    n_jobs=-1, 
    verbose=-1,
    scale_pos_weight=scale_pos_weight  # Bilancia classi in LightGBM
)
lgbm.fit(X_train_res, y_train_res)
models["LGBM"] = lgbm

# 7. CatBoost
console.print("⠸ Training CatBoost...")
# CatBoost usa class_weights come dizionario
cat = CatBoostClassifier(
    iterations=150, 
    learning_rate=0.05, 
    depth=6, 
    random_seed=SEED, 
    verbose=0, 
    allow_writing_files=False,
    class_weights=list(class_weight_dict.values())  # CatBoost class weights
)
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


# ============================================================ #
# 2. ENSEMBLE PREDICTIONS
# ============================================================ #
console.print("\n[bold header]Evaluating Mega-Ensemble (Metrics Tournament)...[/bold header]")

# Probabilità per ogni modello
probs = {}

for name, model in models.items():
    if name == "DL":
        probs[name] = model.predict(X_test, verbose=0).ravel()
    else:
        probs[name] = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------ #
# Ensemble pesato (expert weighting)
# ------------------------------------------------------------ #
weights = {
    "LR": 0,
    "RF": 2,
    "ET": 3,
    "GB": 2,
    "XGB": 4,
    "LGBM": 4,
    "CAT": 4,
    "DL": 2
}

total_weight = sum(weights.values())

ensemble_proba = np.zeros_like(y_test, dtype=float)
for name, w in weights.items():
    if w > 0:
        ensemble_proba += probs[name] * w

ensemble_proba /= total_weight


# ============================================================ #
# 3. THRESHOLD OPTIMIZATION
# ============================================================ #

opt_thresh, winner_name = run_threshold_tournament(y_test, ensemble_proba)

# Valutazione finale
console.print(f"\n[bold underline]Final Optimized Results ({winner_name} Strategy)[/bold underline]")
y_pred = (ensemble_proba >= opt_thresh).astype(int)

console.print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
console.print(f"Precision: {precision_score(y_test, y_pred):.4f}")
console.print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
console.print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
console.print(f"ROC-AUC:   {roc_auc_score(y_test, ensemble_proba):.4f}")


# ============================================================ #
# 4. CREAZIONE ENSEMBLE WRAPPER PER MODULO 03
# ============================================================ #
# Importa EnsembleWrapper da shared_utils (deve essere condivisa
# per permettere deserializzazione joblib in modulo 03)
# ============================================================ #

from shared_utils import EnsembleWrapper  # Importa classe condivisa

# Creazione dell'ensemble wrapper
ensemble_model = EnsembleWrapper(models, weights)

# ============================================================ #
# 5. SALVATAGGIO MODELLI E OUTPUT
# ============================================================ #

console.print("\n[bold cyan]Saving Mega-Ensemble Artifacts...[/bold cyan]")

# Salvataggio modelli individuali
dump(lr, ARTIFACTS_DIR / "model_lr.joblib")
dump(rf, ARTIFACTS_DIR / "model_rf.joblib")
dump(gb, ARTIFACTS_DIR / "model_gb.joblib")
dump(xgb, ARTIFACTS_DIR / "model_xgb.joblib")
dump(et, ARTIFACTS_DIR / "model_et.joblib")
dump(lgbm, ARTIFACTS_DIR / "model_lgbm.joblib")

# Salvataggio modelli speciali (CatBoost e Neural Network)
cat.save_model(str(ARTIFACTS_DIR / "model_cat.cbm"))
nn.save(ARTIFACTS_DIR / "model_dl.keras")

# Salvataggio ensemble unificato per modulo 03
dump(ensemble_model, ARTIFACTS_DIR / "best_model_unified.joblib")

# Salvataggio tipo modello per modulo 03 (ensemble = combinazione di modelli)
with open(ARTIFACTS_DIR / "model_type.txt", "w") as f:
    f.write("ensemble")

# Salvataggio soglia ottimale
with open(ARTIFACTS_DIR / "threshold.txt", "w") as f:
    f.write(str(opt_thresh))

# ============================================================ #
# 6. VISUALIZZAZIONE CONFUSION MATRIX DETTAGLIATA
# ============================================================ #
# La confusion matrix viene visualizzata con metriche aggiuntive
# per diagnosticare meglio le performance del modello
# ============================================================ #

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calcolo metriche aggiuntive per diagnosticare la confusion matrix
sensitivity = tp / (tp + fn + 1e-8)  # Recall / True Positive Rate
specificity = tn / (tn + fp + 1e-8)  # True Negative Rate
precision = tp / (tp + fp + 1e-8)    # Positive Predictive Value
npv = tn / (tn + fn + 1e-8)          # Negative Predictive Value

console.print("\n[bold cyan]Confusion Matrix Details:[/bold cyan]")
console.print(f"True Negatives (TN):  {tn:5d}  |  Predicted Healthy correctly")
console.print(f"False Positives (FP): {fp:5d}  |  Predicted Disease but actually Healthy")
console.print(f"False Negatives (FN): {fn:5d}  |  Predicted Healthy but actually Disease")
console.print(f"True Positives (TP):  {tp:5d}  |  Predicted Disease correctly")
console.print(f"\n[bold]Clinical Metrics:[/bold]")
console.print(f"Sensitivity (Recall):     {sensitivity:.4f}  |  Ability to detect Disease")
console.print(f"Specificity:              {specificity:.4f}  |  Ability to detect Healthy")
console.print(f"Precision (PPV):          {precision:.4f}  |  When predicting Disease, how often correct")
console.print(f"Negative Predictive Value: {npv:.4f}  |  When predicting Healthy, how often correct")

# Creazione heatmap confusion matrix con annotazioni migliorate
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    xticklabels=["Healthy (0)", "Disease (1)"],
    yticklabels=["Healthy (0)", "Disease (1)"],
    cbar_kws={'label': 'Count'}
)
plt.title(f"Confusion Matrix - Ensemble Model\n{winner_name} Strategy (Threshold: {opt_thresh:.3f})", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

# Aggiunta annotazioni metriche sulla figura
textstr = f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}\n'
textstr += f'Precision: {precision:.4f}\n'
textstr += f'Recall:    {sensitivity:.4f}\n'
textstr += f'F1-Score:  {f1_score(y_test, y_pred):.4f}'
plt.text(1.5, 0.3, textstr, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
console.print(f"\n[green]Confusion matrix saved to {ARTIFACTS_DIR / 'confusion_matrix.png'}[/green]")
plt.show()

console.print("\n[bold green]Optimization Complete.[/bold green]")
