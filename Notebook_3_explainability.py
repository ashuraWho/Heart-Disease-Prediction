# ============================================================
# NOTEBOOK 3 – EXPLAINABILITY & INTERPRETABILITÀ
# Heart Disease Prediction
# ============================================================
#
# OBIETTIVO:
# - Capire come e perché i modelli prendono decisioni
# - Rendere il modello interpretabile in ambito medico
# - Identificare i fattori di rischio principali
#
# Tecniche usate:
# - Coefficienti della Logistic Regression
# - Feature Importance (Random Forest)
# - SHAP values (globali e locali)
#
# ============================================================

# =====================
# 1. IMPORT LIBRARIES
# =====================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modelli (già addestrati nel Notebook 2)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Explainability
import shap

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =====================
# 2. ASSUNZIONI
# =====================
# Questo notebook assume che siano già disponibili:
# - X_train_processed
# - X_test_processed
# - y_train, y_test
# - preprocessor (ColumnTransformer)
# - best_rf (Random Forest ottimizzato)
# - log_reg (Logistic Regression)

import Notebook_1_eda as eda
import Notebook_2_ML as ml

# Recuperiamo i nomi delle feature dopo One-Hot Encoding

feature_names_num = eda.preprocessor.transformers_[0][2]
feature_names_cat = eda.preprocessor.transformers_[1][1] \
    .named_steps['onehot'] \
    .get_feature_names_out(eda.preprocessor.transformers_[1][2])

feature_names = np.concatenate([
    feature_names_num,
    feature_names_cat
])

# =====================
# 3. LOGISTIC REGRESSION – COEFFICIENTI
# =====================
# I coefficienti indicano l'impatto (positivo/negativo)
# di ogni feature sul log-odds della malattia

log_reg_coef = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': ml.log_reg.coef_[0]
})

# Ordiniamo per importanza assoluta
log_reg_coef['AbsCoeff'] = log_reg_coef['Coefficient'].abs()
log_reg_coef = log_reg_coef.sort_values('AbsCoeff', ascending=False)

# Visualizzazione dei top 15 coefficienti
sns.barplot(
    x='Coefficient',
    y='Feature',
    data=log_reg_coef.head(15)
)
plt.title('Top 15 Feature – Logistic Regression')
plt.show()

# COMMENTO:
# - Coefficienti positivi -> aumentano il rischio
# - Coefficienti negativi -> effetto protettivo
# - Modello lineare = facile da spiegare ai clinici

# =====================
# 4. RANDOM FOREST – FEATURE IMPORTANCE
# =====================
# Le Random Forest misurano l'importanza media
# delle feature negli split degli alberi

rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': ml.best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualizziamo le top 15 feature
sns.barplot(
    x='Importance',
    y='Feature',
    data=rf_importance.head(15)
)
plt.title('Top 15 Feature – Random Forest')
plt.show()

# COMMENTO:
# - Modello non lineare
# - Cattura interazioni complesse
# - Meno interpretabile della LogReg, ma più potente

# =====================
# 5. SHAP – INTRODUZIONE
# =====================
# SHAP (SHapley Additive exPlanations):
# - Basato su teoria dei giochi
# - Spiegazioni locali e globali
# - Stato dell'arte per explainability

# =====================
# 6. SHAP – RANDOM FOREST (GLOBAL)
# =====================
# Convertiamo in array denso
X_test_dense = eda.X_test_processed

# Creiamo l'explainer per modelli ad albero
explainer_rf = shap.TreeExplainer(ml.best_rf)

# Calcoliamo SHAP values sul test set
shap_values_rf = explainer_rf.shap_values(eda.X_test_processed)

# Summary plot (importanza globale)
shap.summary_plot(
    shap_values_rf[1],
    eda.X_test_processed,
    feature_names=feature_names
)

# COMMENTO:
# - Ogni punto è un paziente
# - Colore = valore della feature
# - Asse X = impatto sulla predizione

# =====================
# 7. SHAP – BAR PLOT (GLOBAL)
# =====================

shap.summary_plot(
    shap_values_rf[1],
    eda.X_test_processed,
    feature_names=feature_names,
    plot_type='bar'
)

# =====================
# 8. SHAP – LOCAL EXPLANATION (SINGOLO PAZIENTE)
# =====================
# Analizziamo una singola predizione

patient_idx = 0  # primo paziente del test set

shap.force_plot(
    explainer_rf.expected_value[1],
    shap_values_rf[1][patient_idx],
    eda.X_test_processed[patient_idx],
    feature_names=feature_names,
    matplotlib=True
)

# COMMENTO:
# - Rosso -> aumenta il rischio
# - Blu -> riduce il rischio
# - Utile per spiegazioni cliniche individuali

# =====================
# 9. CONFRONTO MODELLI (INTERPRETABILITÀ)
# =====================
# Logistic Regression:
# - + Molto interpretabile
# - - Assume relazioni lineari

# Random Forest:
# - + Migliori performance
# - - Meno trasparente

# SHAP:
# - Ponte tra performance e interpretabilità