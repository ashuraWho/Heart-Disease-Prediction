# ============================================================
# NOTEBOOK 2 – MACHINE LEARNING CLASSICO + HYPERPARAMETER TUNING
# Heart Disease Prediction
# ============================================================
#
# OBIETTIVO:
# - Costruire modelli di ML supervisionato
# - Definire una baseline
# - Confrontare diversi algoritmi
# - Ottimizzare gli iperparametri
# - Valutare le metriche clinicamente rilevanti
#
# NOTA IMPORTANTE:
# - Usiamo i dati preprocessati dal Notebook 1
# ============================================================

# =====================
# 1. IMPORT LIBRARIES
# =====================

# Manipolazione dati
import numpy as np
import pandas as pd

# Modelli ML
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Valutazione modelli
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

# Cross-validation e tuning
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Visualizzazione
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# =====================
# 2. LOAD PREPROCESSED DATA
# =====================
# NOTA:
# Questo notebook assume che le seguenti variabili siano
# già presenti in memoria (dal Notebook 1):
# - X_train_processed
# - X_test_processed
# - y_train
# - y_test

# Se lavori in un notebook separato, puoi:
# - salvare i dati preprocessati con joblib
# - oppure rieseguire Notebook 1

import Notebook_1_eda as eda

# =====================
# 3. METRICHE DI VALUTAZIONE
# =====================
# In ambito medico la accuracy non è sufficiente.
# Definiamo una funzione di valutazione completa.


def evaluate_model(model, X_test, y_test):
    """
    Valuta un modello di classificazione binaria
    usando metriche clinicamente rilevanti.
    """

    # Predizioni
    y_pred = model.predict(X_test)

    # Probabilità (necessarie per ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    # Calcolo metriche
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

    return results

# =====================
# 4. BASELINE MODEL
# =====================
# DummyClassifier: predice sempre la classe più frequente
# Serve come riferimento minimo

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(eda.X_train_processed, eda.y_train)

baseline_results = evaluate_model(
    baseline,
    eda.X_test_processed,
    eda.y_test
)

print("Baseline Results:")
print(baseline_results)

# COMMENTO:
# Ogni modello deve battere questa baseline

# =====================
# 5. LOGISTIC REGRESSION
# =====================
# Modello lineare interpretabile

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # gestisce lo sbilanciamento
    solver='liblinear'
)

log_reg.fit(eda.X_train_processed, eda.y_train)

log_reg_results = evaluate_model(
    log_reg,
    eda.X_test_processed,
    eda.y_test
)

print("\nLogistic Regression Results:")
print(log_reg_results)

# =====================
# 6. K-NEAREST NEIGHBORS
# =====================
# Modello non parametrico
# Sensibile allo scaling (già fatto)

knn = KNeighborsClassifier()

knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
}

knn_cv = GridSearchCV(
    knn,
    knn_params,
    cv=5,
    scoring='recall',  # privilegiamo la sensibilità
    n_jobs=-1
)

knn_cv.fit(eda.X_train_processed, eda.y_train)

best_knn = knn_cv.best_estimator_

knn_results = evaluate_model(
    best_knn,
    eda.X_test_processed,
    eda.y_test
)

print("\nBest KNN Parameters:", knn_cv.best_params_)
print("KNN Results:")
print(knn_results)

# =====================
# 7. SUPPORT VECTOR MACHINE (RBF)
# =====================
# Modello potente su dataset piccoli
# Richiede tuning accurato

svm = SVC(probability=True, class_weight='balanced')

svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1]
}

svm_cv = GridSearchCV(
    svm,
    svm_params,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

svm_cv.fit(eda.X_train_processed, eda.y_train)

best_svm = svm_cv.best_estimator_

svm_results = evaluate_model(
    best_svm,
    eda.X_test_processed,
    eda.y_test
)

print("\nBest SVM Parameters:", svm_cv.best_params_)
print("SVM Results:")
print(svm_results)

# =====================
# 8. RANDOM FOREST
# =====================
# Modello ad alberi robusto e non lineare

rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

rf_cv = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

rf_cv.fit(eda.X_train_processed, eda.y_train)

best_rf = rf_cv.best_estimator_

rf_results = evaluate_model(
    best_rf,
    eda.X_test_processed,
    eda.y_test
)

print("\nBest Random Forest Parameters:", rf_cv.best_params_)
print("Random Forest Results:")
print(rf_results)

# =====================
# 9. ROC CURVES COMPARISON
# =====================

plt.figure(figsize=(8, 6))

models = {
    'Logistic Regression': log_reg,
    'KNN': best_knn,
    'SVM': best_svm,
    'Random Forest': best_rf
}

for name, model in models.items():
    RocCurveDisplay.from_estimator(
        model,
        eda.X_test_processed,
        eda.y_test,
        name=name,
        ax=plt.gca()
    )

plt.title('ROC Curve Comparison')
plt.show()

# =====================
# 10. CONFUSION MATRIX (BEST MODEL)
# =====================
# Selezioniamo il modello con recall più alto

results_df = pd.DataFrame([
    log_reg_results,
    knn_results,
    svm_results,
    rf_results
], index=['LogReg', 'KNN', 'SVM', 'RF'])

print("\nSummary Results:\n")
print(results_df)

best_model_name = results_df['Recall'].idxmax()
print(f"\nBest model based on Recall: {best_model_name}")

best_model = models[{
    'LogReg': 'Logistic Regression',
    'KNN': 'KNN',
    'SVM': 'SVM',
    'RF': 'Random Forest'
}[best_model_name]]

# Confusion Matrix
y_pred_best = best_model.predict(eda.X_test_processed)
cm = confusion_matrix(eda.y_test, y_pred_best)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix – {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()