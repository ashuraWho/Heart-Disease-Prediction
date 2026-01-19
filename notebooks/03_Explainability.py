# ============================================================ #
# Module 03 – Explainability & Interpretability                #
# ============================================================ #
# Questo modulo:
# - carica il modello migliore addestrato (ML o DL)
# - ricostruisce i nomi delle feature post-preprocessing
# - fornisce interpretabilità globale (feature importance)
# - genera spiegazioni SHAP globali e locali
# - permette di spiegare singole predizioni (patient-level)
# ============================================================ #

import sys
from pathlib import Path

# ------------------------------------------------------------ #
# Aggiunta del project root al PYTHONPATH
# Necessario per importare shared_utils
# ------------------------------------------------------------ #
sys.path.append(str(Path(__file__).resolve().parent))

try:
    # Utility condivise
    from shared_utils import (
        setup_environment, # Setup seed + env vars
        console, # Logging strutturato
        ARTIFACTS_DIR, # Directory artefatti
        EnsembleWrapper # Classe wrapper ensemble (necessaria per deserializzazione joblib)
    )
except ImportError:
    print("Error: shared_utils not found.")
    sys.exit(1)

# ------------------------------------------------------------ #
# Setup globale dell'ambiente
# ------------------------------------------------------------ #
setup_environment()

# ------------------------------------------------------------ #
# Import scientific stack
# ------------------------------------------------------------ #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from joblib import load
import tensorflow as tf


# ============================================================ #
# CONFIGURAZIONE VISIVA
# ============================================================ #
sns.set(style="whitegrid") # Stile uniforme dei grafici
plt.rcParams["figure.figsize"] = (10, 6) # Dimensione default


# ============================================================ #
# CARICAMENTO ARTEFATTI
# ============================================================ #
# Carica:
# - preprocessor per ricostruire feature names
# - modello unificato (ensemble o Keras)
# - dati di test per SHAP analysis
# ============================================================ #

console.print("[bold cyan]Loading artifacts for explainability...[/bold cyan]")

try:
    # Preprocessor fittato (necessario per ricostruire feature names)
    preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib")
    
    # Tipo di modello vincente (ensemble, ML classico o Keras)
    model_type = "ensemble"  # Default a ensemble
    try:
        with open(ARTIFACTS_DIR / "model_type.txt", "r") as f:
            model_type = f.read().strip()
    except FileNotFoundError:
        # Se non esiste model_type.txt, assume ensemble
        console.print("[yellow]Warning: model_type.txt not found, assuming ensemble[/yellow]")
    
    # Caricamento modello in base al tipo
    if model_type == "keras":
        model = tf.keras.models.load_model(ARTIFACTS_DIR / "best_model_unified.keras")
    else:
        # Carica ensemble wrapper o modello classico
        model = load(ARTIFACTS_DIR / "best_model_unified.joblib")

    # Dati di test preprocessati
    X_test = np.load(ARTIFACTS_DIR / "X_test.npz")["X"]
    y_test = np.load(ARTIFACTS_DIR / "y_test.npy")

except Exception as e:
    # Catch robusto per mismatch di artefatti/versioni
    console.print(f"[bold red]ERROR: Compatibility issue or missing files: {e}[/bold red]")
    console.print("[yellow]>>> FIX: Please run 'Module 01' and 'Module 02' again.[/yellow]")
    sys.exit(1)

console.print(f"[green]Artifacts loaded successfully. Active Model Type:[/green] [bold]{model_type}[/bold]")


# ============================================================ #
# RICOSTRUZIONE NOMI FEATURE
# ============================================================ #
# Necessario perché dopo OneHotEncoding le feature
# non corrispondono più alle colonne originali
# ============================================================ #

feature_names = preprocessor.get_feature_names_out()


# ============================================================ #
# INTERPRETABILITÀ GLOBALE (NON-SHAP)
# ============================================================ #
# Visualizza feature importance basata sul tipo di modello.
# Per ensemble, usa l'importanza media dei modelli ad albero.
# ============================================================ #

console.print("\n[bold header]Global Feature Importance[/bold header]")

# ------------------------------------------------------------ #
# Caso 1: Ensemble Wrapper
# ------------------------------------------------------------ #
if model_type == "ensemble" and hasattr(model, "models"):
    # Per ensemble, calcola importanza media dai modelli ad albero
    console.print("[cyan]Calculating average feature importance from tree models in ensemble...[/cyan]")
    
    importances_list = []
    tree_models = ["RF", "ET", "GB", "XGB", "LGBM", "CAT"]
    
    for name in tree_models:
        if name in model.models and hasattr(model.models[name], "feature_importances_"):
            importances_list.append(model.models[name].feature_importances_)
    
    if importances_list:
        # Media delle importanze
        avg_importance = np.mean(importances_list, axis=0)
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": avg_importance
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=imp_df.head(20))
        plt.title("Top 20 Features – Ensemble Average Importance (Tree Models)", fontsize=14, fontweight='bold')
        plt.xlabel("Average Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------ #
# Caso 2: Modello Lineare (Logistic Regression)
# ------------------------------------------------------------ #
elif hasattr(model, "coef_"):
    
    # Costruzione DataFrame coefficiente-feature
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })
    
    # Valore assoluto per ranking
    coef_df["AbsCoeff"] = coef_df["Coefficient"].abs()
    
    # Ordinamento per importanza
    coef_df = coef_df.sort_values("AbsCoeff", ascending=False)

    # Plot delle top feature
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df.head(20))
    plt.title("Top 20 Features – Linear Model Coefficients", fontsize=14, fontweight='bold')
    plt.xlabel("Coefficient", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------ #
# Caso 3: Modelli ad Alberi (RF, GB, XGB, LGBM, CAT)
# ------------------------------------------------------------ #
elif hasattr(model, "feature_importances_"):
    # Tree Model
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=imp_df.head(20))
    plt.title("Top 20 Features – Tree Model Importance", fontsize=14, fontweight='bold')
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

else:
    console.print("[yellow]Warning: Model type does not support feature importance visualization[/yellow]")


# ============================================================ #
# SHAP – EXPLAINABILITY AVANZATA
# ============================================================ #

console.print("\n[bold header]Generating SHAP Explanations...[/bold header]")

# ------------------------------------------------------------ #
# Inizializzazione Explainer
# ------------------------------------------------------------ #
with console.status("Initializing SHAP explainer...", spinner="dots"):
    if model_type == "keras":
        # ---------------------------------------------------- #
        # KernelExplainer per Deep Learning:
        # - modello black-box
        # - molto costoso computazionalmente
        # - background set ridotto per performance
        # ---------------------------------------------------- #
        explainer = shap.KernelExplainer(model.predict, X_test[:20], verbose=0)
        shap_values = explainer.shap_values(X_test[:10])
        shap_vals_raw = shap_values[0] if isinstance(shap_values, list) else shap_values
        X_test_shap = X_test[:10]  # Usa solo un subset per visualizzazione
    
    elif model_type == "ensemble":
        # ---------------------------------------------------- #
        # Caso Ensemble: usa KernelExplainer per ensemble wrapper
        # perché è un modello composito
        # ---------------------------------------------------- #
        console.print("[cyan]Using KernelExplainer for ensemble model...[/cyan]")
        explainer = shap.KernelExplainer(model.predict, X_test[:50], verbose=0)
        shap_values = explainer.shap_values(X_test[:100])  # Sample più grande ma non tutto
        shap_vals_raw = shap_values[0] if isinstance(shap_values, list) else shap_values
        X_test_shap = X_test[:100]
    
    else:
        # ---------------------------------------------------- #
        # Caso ML classico singolo modello
        # ---------------------------------------------------- #
        if hasattr(model, "feature_importances_"):
            # TreeExplainer: veloce e preciso per modelli ad albero
            console.print("[cyan]Using TreeExplainer for tree-based model...[/cyan]")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test)
        else:
            # Explainer generico per modelli lineari
            console.print("[cyan]Using generic Explainer for linear model...[/cyan]")
            explainer = shap.Explainer(model.predict_proba, X_test[:100])
            shap_values = explainer(X_test[:100])
        
        # Normalizzazione output SHAP
        shap_vals_raw = shap_values.values if hasattr(shap_values, "values") else shap_values
        X_test_shap = X_test
        
        # Caso classificazione binaria probabilistica
        if len(shap_vals_raw.shape) == 3: 
            shap_vals_raw = shap_vals_raw[:, :, 1]


# ============================================================ #
# SHAP SUMMARY PLOT (GLOBAL)
# ============================================================ #
# Visualizza importanza globale delle feature basata su SHAP values
# ============================================================ #

console.print("\n[bold cyan]Generating SHAP Summary Plot...[/bold cyan]")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_vals_raw,
    X_test_shap,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=300, bbox_inches='tight')
console.print(f"[green]SHAP summary plot saved to {ARTIFACTS_DIR / 'shap_summary.png'}[/green]")
plt.show()


# ============================================================ #
# SPIEGAZIONE LOCALE (PATIENT-LEVEL)
# ============================================================ #
# Mostra l'impatto di ogni feature sulla predizione di un singolo paziente
# ============================================================ #

patient_idx = 0  # Indice paziente
console.print(f"\n[cyan]Explaining prediction for Patient #{patient_idx}...[/cyan]")

# ------------------------------------------------------------ #
# Caso SHAP values strutturati (Tree / Linear)
# ------------------------------------------------------------ #
if hasattr(shap_values, "values") and model_type not in ["keras", "ensemble"]:
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[patient_idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall Plot - Patient #{patient_idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_waterfall.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]SHAP waterfall plot saved to {ARTIFACTS_DIR / 'shap_waterfall.png'}[/green]")
    plt.show()
    
# ------------------------------------------------------------ #
# Caso Kernel SHAP (Keras / Ensemble)
# ------------------------------------------------------------ #
else:
    # Estrai SHAP values per il paziente specificato (assicura forma 1D)
    patient_shap = shap_vals_raw[patient_idx]
    if len(patient_shap.shape) > 1:  # Se è 2D, flatten
        patient_shap = patient_shap.flatten()
    
    # Prendi top 15 feature per valore assoluto SHAP
    abs_shap = np.abs(patient_shap)
    top_indices = np.argsort(abs_shap)[-15:][::-1]  # Top 15 in ordine decrescente
    
    # Plot manuale barh (più affidabile per array numpy)
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_indices))
    bar_colors = ['red' if patient_shap[i] < 0 else 'blue' for i in top_indices]
    plt.barh(y_pos, patient_shap[top_indices], color=bar_colors)
    plt.yticks(y_pos, [feature_names[i] for i in top_indices])
    plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f"SHAP Bar Plot - Patient #{patient_idx}\nBlue=Positive impact, Red=Negative impact", fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_bar.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]SHAP bar plot saved to {ARTIFACTS_DIR / 'shap_bar.png'}[/green]")
    plt.show()
