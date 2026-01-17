# ============================================================ # Module 02 – Classical Machine Learning
# MODULE 02 – CLASSICAL MACHINE LEARNING                       # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
import sys # Import sys to check the Python environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda

# --- DIAGNOSTIC: Check Environment ---
print(f"Python Executable: {sys.executable}") # Print the path to the current Python interpreter
print(f"Python Version: {sys.version}") # Print the version of Python being used
if "anaconda3/bin/python" in sys.executable or "miniconda3/bin/python" in sys.executable: # Check if running in base
    print("WARNING: You are likely running in the 'base' environment. This is discouraged.") # Warn the user
# -------------------------------------

from pathlib import Path # Import Path for robust filesystem path manipulation
import numpy as np # Import numpy for numerical operations and array handling
import pandas as pd # Import pandas for data manipulation and tabular data

import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for statistical data visualization
sns.set(style="whitegrid") # Set a clean, whitegrid plotting style for seaborn

from sklearn.dummy import DummyClassifier # Import DummyClassifier for baseline comparison
from sklearn.linear_model import LogisticRegression # Import LogisticRegression for classification
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier for k-NN model
from sklearn.svm import SVC # Import SVC for Support Vector Classification
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier for ensemble learning

from sklearn.model_selection import GridSearchCV, StratifiedKFold # Import tools for cross-validation and hyperparameter tuning
from sklearn.metrics import ( # Import various classification metrics
    accuracy_score, precision_score, recall_score, # Import Accuracy, Precision, Recall
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay # Import F1, ROC-AUC, CM, and ROC Curve display
) # End of metrics import

from joblib import load, dump # Import load and dump from joblib for object serialization

# ===================== # Header Section
# PATHS                 # Paths Header
# ===================== # Header Section

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Define the root directory of the project
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

# ===================== # Header Section
# LOAD ARTIFACTS        # Load Artifacts Header
# ===================== # Header Section

try: # Start safety block for library version compatibility
    X_train = np.load(ARTIFACTS_DIR / "X_train.npz")["X"] # Load preprocessed training features
    X_test  = np.load(ARTIFACTS_DIR / "X_test.npz")["X"] # Load preprocessed testing features
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy") # Load training labels
    y_test  = np.load(ARTIFACTS_DIR / "y_test.npy") # Load testing labels
except Exception as e: # Catch loading errors
    print(f"ERROR: Could not load artifacts: {e}") # Print error
    print(">>> Please run Module 01 first.") # Print fix
    sys.exit(1) # Exit

print("Artifacts loaded successfully") # Print confirmation message for artifact loading

# ===================== # Header Section
# EVALUATION FUNCTION   # Evaluation Function Header
# ===================== # Header Section

def evaluate_model(model, X_test, y_test): # Define a function to evaluate model performance
    y_pred = model.predict(X_test) # Generate class predictions for the test set

    if hasattr(model, "predict_proba"): # Check if model can predict probabilities
        y_proba = model.predict_proba(X_test)[:, 1] # Extract probabilities for the positive class
    else: # If model doesn't support predict_proba
        y_proba = model.decision_function(X_test) # Use decision function scores instead

    cm = confusion_matrix(y_test, y_pred) # Calculate confusion matrix
    tn, fp, fn, tp = cm.ravel() # Extract matrix components to diagnose False Positives (FP)

    return { # Return a dictionary of performance metrics
        "Accuracy": accuracy_score(y_test, y_pred), # Calculate Accuracy
        "Precision": precision_score(y_test, y_pred, zero_division=0), # Calculate Precision (focus on FP)
        "Recall": recall_score(y_test, y_pred, zero_division=0), # Calculate Recall (focus on FN)
        "F1": f1_score(y_test, y_pred, zero_division=0), # Calculate F1-Score (balance)
        "ROC-AUC": roc_auc_score(y_test, y_proba), # Calculate ROC-AUC Score
        "FP": fp, # Return count of False Positives for diagnostics
        "FN": fn # Return count of False Negatives for diagnostics
    } # End of dictionary

# ===================== # Header Section
# BASELINE             # Baseline Header
# ===================== # Header Section

baseline = DummyClassifier(strategy="most_frequent") # Initialize a baseline model that always predicts the most frequent class
baseline.fit(X_train, y_train) # Fit the baseline model on training data
baseline_results = evaluate_model(baseline, X_test, y_test) # Evaluate the baseline model on test data

# ===================== # Header Section
# CV STRATEGY           # Cross-Validation Strategy Header
# ===================== # Header Section

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Define a 5-fold stratified cross-validation strategy

# ===================== # Header Section
# MODELS                # Models Header
# ===================== # Header Section

models = { # Define a dictionary containing models and their hyperparameter grids
    "Logistic Regression": ( # Logistic Regression configuration
        LogisticRegression( # Initialize Logistic Regression model
            max_iter=1000, # Set maximum iterations for convergence
            class_weight="balanced", # Use balanced class weights to handle imbalance
            solver="liblinear" # Specify solver for small datasets
        ), # End of model init
        {"C": [0.01, 0.1, 1, 10]} # Hyperparameter grid for 'C' (inverse regularization strength)
    ), # End of LR config
    "KNN": ( # K-Nearest Neighbors configuration
        KNeighborsClassifier(), # Initialize KNN model
        {"n_neighbors": [3, 5, 7, 9, 11]} # Hyperparameter grid for number of neighbors
    ), # End of KNN config
    "SVM": ( # Support Vector Machine configuration
        SVC(probability=True, class_weight="balanced"), # Initialize SVM with probability estimates and balanced weights
        {"C": [0.1, 1, 10]} # Hyperparameter grid for 'C'
    ), # End of SVM config
    "Random Forest": ( # Random Forest configuration
        RandomForestClassifier( # Initialize Random Forest model
            random_state=42, # Set random state for reproducibility
            class_weight="balanced" # Use balanced class weights
        ), # End of model init
        {"n_estimators": [100, 300], "max_depth": [None, 5, 10]} # Hyperparameter grid for number of trees and max depth
    ) # End of RF config
} # End of models dictionary

results = {"Baseline": baseline_results} # Initialize results dictionary with baseline results
best_models = {} # Initialize dictionary to store the best tuned models

# ===================== # Header Section
# TRAINING LOOP         # Training Loop Header
# ===================== # Header Section
# We use 'f1' scoring to better balance Precision and Recall, reducing excessive False Positives.

for name, (model, params) in models.items(): # Iterate through each model and its parameter grid
    gs = GridSearchCV( # Initialize Grid Search Cross-Validation
        model, # The model to tune
        params, # The hyperparameter grid
        cv=cv, # The CV strategy
        scoring="f1", # Optimize for F1-Score (balances FP and FN) as requested to reduce False Positives
        n_jobs=-1 # Use all available CPU cores
    ) # End of GridSearchCV init
    gs.fit(X_train, y_train) # Fit GridSearchCV to training data

    best_models[name] = gs.best_estimator_ # Store the best estimator found
    results[name] = evaluate_model(gs.best_estimator_, X_test, y_test) # Evaluate the best estimator on test data

# ===================== # Header Section
# RESULTS SUMMARY       # Results Summary Header
# ===================== # Header Section

results_df = pd.DataFrame(results).T # Convert the results dictionary to a transposed DataFrame
print("\nFinal Model Comparison:") # Print header for comparison table
print(results_df) # Print the comparison table

# ===================== # Header Section
# BEST MODEL            # Best Model Header
# ===================== # Header Section

best_model_name = results_df["F1"].idxmax() # Identify the model with the highest F1 score
best_model = best_models[best_model_name] # Retrieve the best model instance

print(f"\nBest model by F1-Score: {best_model_name}") # Print the name of the best model

# ===================== # Header Section
# CONFUSION MATRIX      # Confusion Matrix Header
# ===================== # Header Section

cm = confusion_matrix(y_test, best_model.predict(X_test)) # Calculate the confusion matrix for the best model
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") # Visualize the confusion matrix as a heatmap
plt.title(f"Confusion Matrix – {best_model_name}") # Set title for confusion matrix plot
plt.xlabel("Predicted") # Set X axis label
plt.ylabel("Actual") # Set Y axis label
plt.show() # Display the confusion matrix plot

# ===================== # Header Section
# SAVE BEST MODEL       # Save Best Model Header
# ===================== # Header Section

dump(best_model, ARTIFACTS_DIR / "best_model_classic.joblib") # Save the best classical model to artifacts
