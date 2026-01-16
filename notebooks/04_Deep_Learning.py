# ============================================================
# NOTEBOOK 4 – DEEP LEARNING (MULTI-LAYER PERCEPTRON)
# Heart Disease Prediction
# ============================================================
#
# OBIETTIVI:
# - Valutare una MLP su dataset tabulare piccolo
# - Applicare tecniche robuste di regularization
# - Confronto concettuale DL vs ML classico
#
# NOTA ARCHITETTURALE:
# - Questo notebook NON dipende da altri notebook
# - I dati arrivano esclusivamente da /artifacts
# ============================================================


# =====================
# 1. IMPORT LIBRARIES
# =====================

# Riproducibilità
import random
import numpy as np
import tensorflow as tf

# Dati
import pandas as pd

# Visualizzazione
import matplotlib.pyplot as plt
import seaborn as sns

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Metriche
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)

# Stile grafico
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# =====================
# 2. RIPRODUCIBILITÀ
# =====================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================
# 3. LOAD DATA (ARTIFACTS)
# =====================
# I dati sono stati preprocessati e salvati
# nel Notebook 1 come artifacts versionati

X_train = np.load("../artifacts/X_train.npz")["X"]
X_test  = np.load("../artifacts/X_test.npz")["X"]

y_train = np.load("../artifacts/y_train.npy")
y_test  = np.load("../artifacts/y_test.npy")

input_dim = X_train.shape[1]


# =====================
# 4. MODEL ARCHITECTURE
# =====================
# MLP volutamente compatta:
# - pochi layer
# - forte regolarizzazione
# - adatta a dati tabulari piccoli

model = Sequential([

    Dense(
        32,
        activation="relu",
        input_shape=(input_dim,),
        kernel_regularizer=l2(1e-3)
    ),
    BatchNormalization(),
    Dropout(0.4),

    Dense(
        16,
        activation="relu",
        kernel_regularizer=l2(1e-3)
    ),
    BatchNormalization(),
    Dropout(0.3),

    Dense(
        8,
        activation="relu",
        kernel_regularizer=l2(1e-3)
    ),

    Dense(1, activation="sigmoid")
])


# =====================
# 5. COMPILATION
# =====================

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =====================
# 6. EARLY STOPPING
# =====================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)


# =====================
# 7. TRAINING
# =====================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)


# =====================
# 8. TRAINING DIAGNOSTICS
# =====================

history_df = pd.DataFrame(history.history)

# Loss
plt.plot(history_df["loss"], label="Train")
plt.plot(history_df["val_loss"], label="Validation")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy
plt.plot(history_df["accuracy"], label="Train")
plt.plot(history_df["val_accuracy"], label="Validation")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# =====================
# 9. TEST SET EVALUATION
# =====================

y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

nn_results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
}

print("\nNeural Network Performance:")
for k, v in nn_results.items():
    print(f"{k}: {v:.3f}")


# =====================
# 10. ROC CURVE
# =====================

RocCurveDisplay.from_predictions(
    y_test,
    y_pred_proba,
    name="Neural Network"
)

plt.title("ROC Curve – Neural Network")
plt.show()


# =====================
# 11. CONFUSION MATRIX
# =====================

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix – Neural Network")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =====================
# 12. SAVE MODEL
# =====================
# Modello salvato come artifact versionabile

model.save("../artifacts/nn_model.keras")


# =====================
# 13. CONCLUSIONI
# =====================
# - Su dataset piccoli tabulari, DL ≠ soluzione migliore
# - ML classico spesso:
#   • generalizza meglio
#   • è più interpretabile
#   • costa meno in termini computazionali
#
# Questo notebook dimostra consapevolezza metodologica,
# non uso ingenuo del Deep Learning.