# ============================================================ #
# Module 03 – Explainability & Interpretability                #
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from joblib import load
import tensorflow as tf

# --- CONFIG ---
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --- LOAD ARTIFACTS ---
console.print("[bold cyan]Loading artifacts for explainability...[/bold cyan]")

try:
    preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib")
    
    with open(ARTIFACTS_DIR / "model_type.txt", "r") as f:
        model_type = f.read().strip()
        
    if model_type == "keras":
        model = tf.keras.models.load_model(ARTIFACTS_DIR / "best_model_unified.keras")
    else:
        model = load(ARTIFACTS_DIR / "best_model_unified.joblib")

    X_test = np.load(ARTIFACTS_DIR / "X_test.npz")["X"]
    y_test = np.load(ARTIFACTS_DIR / "y_test.npy")

except Exception as e:
    console.print(f"[bold red]ERROR: Compatibility issue or missing files: {e}[/bold red]")
    console.print("[yellow]>>> FIX: Please run 'Module 01' and 'Module 02' again.[/yellow]")
    sys.exit(1)

console.print(f"[green]Artifacts loaded successfully. Active Model Type:[/green] [bold]{model_type}[/bold]")

# --- FEATURE NAMES ---
feature_names = preprocessor.get_feature_names_out()

# --- GLOBAL INTERPRETABILITY ---
console.print("\n[bold header]Global Feature Importance[/bold header]")

if hasattr(model, "coef_"):
    # Linear Model
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })
    coef_df["AbsCoeff"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsCoeff", ascending=False)

    sns.barplot(x="Coefficient", y="Feature", data=coef_df.head(15))
    plt.title("Top 15 Features – Linear Model Importance")
    plt.show()

elif hasattr(model, "feature_importances_"):
    # Tree Model
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    sns.barplot(x="Importance", y="Feature", data=imp_df.head(15))
    plt.title("Top 15 Features – Tree Model Importance")
    plt.show()

# --- SHAP EXPLAINABILITY ---
console.print("\n[bold header]Generating SHAP Explanations...[/bold header]")

# Init Explainer
with console.status("Initializing SHAP explainer...", spinner="dots"):
    if model_type == "keras":
        # KernelExplainer for Keras (using small background set for speed)
        explainer = shap.KernelExplainer(model.predict, X_test[:20], verbose=0)
        shap_values = explainer.shap_values(X_test[:10])
        shap_vals_raw = shap_values[0] if isinstance(shap_values, list) else shap_values
    else:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test)
        else:
            explainer = shap.Explainer(model.predict, X_test)
            shap_values = explainer(X_test)
        
        # Normalize
        shap_vals_raw = shap_values.values if hasattr(shap_values, "values") else shap_values
        if len(shap_vals_raw.shape) == 3: 
            shap_vals_raw = shap_vals_raw[:, :, 1]

# Summary Plot
shap.summary_plot(
    shap_vals_raw,
    X_test[:10] if model_type == "keras" else X_test,
    feature_names=feature_names
)

# --- LOCAL EXPLANATION ---
patient_idx = 0
console.print(f"\n[cyan]Explaining prediction for Patient #{patient_idx}...[/cyan]")

if hasattr(shap_values, "values") and model_type != "keras":
    shap.plots.waterfall(shap_values[patient_idx], max_display=15)
else:
    shap.bar_plot(shap_vals_raw[patient_idx], feature_names=feature_names, max_display=15)
