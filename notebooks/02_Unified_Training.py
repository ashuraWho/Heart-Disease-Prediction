# ============================================================ # Module 02 – Unified Model Training & Selection
# NOTEBOOK 2 – ML & DL COMPETITION                            # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
# --- CRITICAL STABILITY & macOS FIXES (MUST BE BEFORE ANY OTHER IMPORTS) ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda (multiple OpenMP runtimes)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN to improve stability on some CPUs
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads to 1 to prevent resource-related SegFaults
os.environ['MKL_NUM_THREADS'] = '1' # Limit MKL threads to prevent binary conflicts in Anaconda
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU execution to bypass potentially unstable GPU/Metal drivers on Mac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TensorFlow logging noise to focus on errors
# ---------------------------------------------------------------------------

print("[TRACE] Initializing system...") # Diagnostic trace
import sys # Import sys to check the Python environment
import random # Import random for reproducibility

print("[TRACE] Loading fundamental Data Science libraries...") # Diagnostic trace
import numpy as np # Import numpy for numerical operations
print("[TRACE] Numpy loaded.") # Diagnostic trace
import pandas as pd # Import pandas for data manipulation
print("[TRACE] Pandas loaded.") # Diagnostic trace
import matplotlib.pyplot as plt # Import matplotlib for plotting
print("[TRACE] Matplotlib loaded.") # Diagnostic trace
import seaborn as sns # Import seaborn for statistical data visualization
print("[TRACE] Seaborn loaded.") # Diagnostic trace

from pathlib import Path # Import Path for robust filesystem path manipulation
from joblib import dump # Import dump from joblib for object serialization

print("[TRACE] Importing Machine Learning tools...") # Diagnostic trace
# Import Scikit-Learn tools                                             # sklearn Imports
from sklearn.linear_model import LogisticRegression # Import Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # Import KNN
from sklearn.svm import SVC # Import SVC
from sklearn.ensemble import RandomForestClassifier # Import Random Forest
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Import CV tools
from sklearn.metrics import ( # Import metrics
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
print("[TRACE] Scikit-Learn tools loaded.") # Diagnostic trace

print("[TRACE] Attempting to load TensorFlow (Critical point)...") # Diagnostic trace
import tensorflow as tf # Import tensorflow for deep learning
print(f"[TRACE] TensorFlow Version: {tf.__version__}") # Diagnostic trace

# Import Keras tools                                                    # Keras Imports
from tensorflow.keras.models import Sequential # Import Sequential model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input # Import layers
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping
from tensorflow.keras.regularizers import l2 # Import L2 regularization
print("[TRACE] Keras tools loaded.") # Diagnostic trace

print("[TRACE] All libraries imported successfully. Proceeding to execution...") # Diagnostic trace

# ===================== # Header Section
# PATHS                 # Paths Header
# ===================== # Header Section

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Define the root directory
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

# ===================== # Header Section
# REPRODUCIBILITY       # Reproducibility Header
# ===================== # Header Section

SEED = 42 # Define global seed
random.seed(SEED) # Set Python seed
np.random.seed(SEED) # Set Numpy seed
tf.random.set_seed(SEED) # Set TensorFlow seed

# ===================== # Header Section
# LOAD DATA             # Load Data Header
# ===================== # Header Section

try: # Safety block for loading
    X_train = np.load(ARTIFACTS_DIR / "X_train.npz")["X"] # Load training features
    X_test  = np.load(ARTIFACTS_DIR / "X_test.npz")["X"] # Load testing features
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy") # Load training labels
    y_test  = np.load(ARTIFACTS_DIR / "y_test.npy") # Load testing labels
except Exception as e: # Catch errors
    print(f"ERROR: Could not load artifacts: {e}") # Print error
    print(">>> Please run Module 01 first.") # Print fix
    sys.exit(1) # Exit

input_dim = X_train.shape[1] # Get number of features
print(f"Data loaded successfully. Input dimensions: {input_dim}") # Confirmation

# ===================== # Header Section
# EVALUATION FUNCTION   # Evaluation Header
# ===================== # Header Section

def evaluate(name, y_true, y_pred, y_proba): # Define generic evaluation function
    f1 = f1_score(y_true, y_pred, zero_division=0) # Calculate F1-Score
    print(f"\n--- {name} Results ---") # Print header
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}") # Print Accuracy
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}") # Print Precision
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}") # Print Recall
    print(f"F1-Score:  {f1:.4f}") # Print F1
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}") # Print ROC-AUC
    return f1 # Return F1 for comparison

# ===================== # Header Section
# 1. CLASSICAL ML       # Classical ML Header
# ===================== # Header Section

print("\n[STEP 1] Tuning Classical Models...") # Status

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) # Define CV strategy

models_to_tune = { # Define candidates
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), {"C": [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED, class_weight="balanced"), {"n_estimators": [100, 200], "max_depth": [5, 10]}),
    "SVM": (SVC(probability=True, class_weight="balanced"), {"C": [0.1, 1, 10]})
} # End dict

best_sklearn_model = None # Init storage
best_sklearn_f1 = -1 # Init storage

for name, (model, params) in models_to_tune.items(): # Loop through candidates
    gs = GridSearchCV(model, params, cv=cv, scoring="f1", n_jobs=-1) # Init GridSearch
    gs.fit(X_train, y_train) # Fit on training data

    y_pred = gs.best_estimator_.predict(X_test) # Predict classes
    y_proba = gs.best_estimator_.predict_proba(X_test)[:, 1] # Predict probabilities

    current_f1 = evaluate(name, y_test, y_pred, y_proba) # Evaluate

    if current_f1 > best_sklearn_f1: # Check if best so far
        best_sklearn_f1 = current_f1 # Update best F1
        best_sklearn_model = gs.best_estimator_ # Update best model
        best_sklearn_name = name # Store name

# ===================== # Header Section
# 2. DEEP LEARNING      # Deep Learning Header
# ===================== # Header Section

print("\n[STEP 2] Training Neural Network (DL)...") # Status

nn_model = Sequential([ # Define MLP architecture
    Input(shape=(input_dim,)), # Input layer
    Dense(32, activation="relu", kernel_regularizer=l2(1e-3)), # Hidden 1
    BatchNormalization(), # Batch Norm
    Dropout(0.3), # Dropout
    Dense(16, activation="relu", kernel_regularizer=l2(1e-3)), # Hidden 2
    Dense(1, activation="sigmoid") # Output layer
]) # End architecture

nn_model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]) # Compile

early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True) # Define callback

nn_model.fit( # Train
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0 # Quiet training
) # End fit

y_proba_nn = nn_model.predict(X_test, verbose=0).ravel() # Get DL probabilities
y_pred_nn = (y_proba_nn >= 0.5).astype(int) # Get DL classes
dl_f1 = evaluate("Deep Learning (MLP)", y_test, y_pred_nn, y_proba_nn) # Evaluate DL

# ===================== # Header Section
# 3. FINAL SELECTION    # Winner Selection Header
# ===================== # Header Section

print("\n[STEP 3] Selecting Final Winner...") # Status

if dl_f1 > best_sklearn_f1: # Check if DL won
    print(f"WINNER: Deep Learning (F1: {dl_f1:.4f})") # Print winner
    nn_model.save(ARTIFACTS_DIR / "best_model_unified.keras") # Save as Keras
    with open(ARTIFACTS_DIR / "model_type.txt", "w") as f: f.write("keras") # Mark type
else: # If classical won
    print(f"WINNER: {best_sklearn_name} (F1: {best_sklearn_f1:.4f})") # Print winner
    dump(best_sklearn_model, ARTIFACTS_DIR / "best_model_unified.joblib") # Save as joblib
    with open(ARTIFACTS_DIR / "model_type.txt", "w") as f: f.write("sklearn") # Mark type

# Final Confusion Matrix for Winner                                     # Winner Confusion Matrix
final_pred = y_pred_nn if dl_f1 > best_sklearn_f1 else best_sklearn_model.predict(X_test) # Get winner preds
cm = confusion_matrix(y_test, final_pred) # Calculate CM
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens") # Plot heatmap
plt.title("Confusion Matrix - Pipeline Winner") # Set title
plt.xlabel("Predicted") # Label X
plt.ylabel("Actual") # Label Y
plt.show() # Show

print("\nExecution complete. Best model saved to artifacts.") # Final status
