# ============================================================
# MODULE 03 – EXPLAINABILITY & INTERPRETABILITY
# Heart Disease Prediction
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

import shap
from joblib import load

# =====================
# PATHS
# =====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# =====================
# LOAD ARTIFACTS
# =====================

preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib")
model = load(ARTIFACTS_DIR / "best_model_classic.joblib")

X_test = np.load(ARTIFACTS_DIR / "X_test.npz")["X"]
y_test = np.load(ARTIFACTS_DIR / "y_test.npy")

print("Artifacts loaded successfully")

# =====================
# FEATURE NAMES
# =====================

num_features = preprocessor.transformers_[0][2]

cat_features = (
    preprocessor
    .transformers_[1][1]
    .named_steps["onehot"]
    .get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
)

feature_names = np.concatenate([num_features, cat_features])

# =====================
# GLOBAL INTERPRETABILITY
# =====================

if hasattr(model, "coef_"):
    # Logistic Regression case
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })
    coef_df["AbsCoeff"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsCoeff", ascending=False)

    sns.barplot(
        x="Coefficient",
        y="Feature",
        data=coef_df.head(15)
    )
    plt.title("Top 15 Features – Logistic Regression")
    plt.show()

elif hasattr(model, "feature_importances_"):
    # Random Forest case
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    sns.barplot(
        x="Importance",
        y="Feature",
        data=imp_df.head(15)
    )
    plt.title("Top 15 Features – Random Forest")
    plt.show()

# =====================
# SHAP EXPLAINABILITY
# =====================

if hasattr(model, "feature_importances_"):
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.Explainer(model, X_test)

shap_values = explainer(X_test)

# =====================
# SHAP SUMMARY
# =====================

shap.summary_plot(
    shap_values.values,
    X_test,
    feature_names=feature_names
)

shap.summary_plot(
    shap_values.values,
    X_test,
    feature_names=feature_names,
    plot_type="bar"
)

# =====================
# LOCAL EXPLANATION
# =====================

patient_idx = 0

shap.plots.waterfall(
    shap_values[patient_idx],
    max_display=15
)