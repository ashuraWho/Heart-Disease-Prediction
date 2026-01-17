# ============================================================ # Module 03 – Explainability & Interpretability
# MODULE 03 – EXPLAINABILITY & INTERPRETABILITY               # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
# --- CRITICAL STABILITY & macOS FIXES (MUST BE BEFORE ANY OTHER IMPORTS) ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1' # Limit MKL threads
# ---------------------------------------------------------------------------

print("Importing system libraries...") # Diagnostic print
import sys # Import sys to check the Python environment
from pathlib import Path # Import Path for robust filesystem path manipulation

print("Importing data science libraries...") # Diagnostic print
import numpy as np # Import numpy for numerical operations and array handling
import pandas as pd # Import pandas for data manipulation and tabular data
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for statistical data visualization

sns.set(style="whitegrid") # Set a clean, whitegrid plotting style for seaborn
plt.rcParams["figure.figsize"] = (10, 6) # Set default figure size for consistency

print("Importing Machine Learning and Explainability tools...") # Diagnostic print
import shap # Import shap for model explainability
from joblib import load # Import load from joblib to retrieve saved models
import tensorflow as tf # Import tensorflow for Keras model loading

print("All libraries imported successfully.") # Diagnostic print

# ===================== # Header Section
# PATHS                 # Paths Header
# ===================== # Header Section

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Define the root directory of the project
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

# ===================== # Header Section
# LOAD ARTIFACTS        # Load Artifacts Header
# ===================== # Header Section

try: # Start safety block for library version compatibility
    preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib") # Load the saved preprocessing pipeline

    # Check model type to load correctly                                # Unified Load Section
    with open(ARTIFACTS_DIR / "model_type.txt", "r") as f: # Open type flag
        model_type = f.read().strip() # Read type (keras or sklearn)

    if model_type == "keras": # If Keras
        model = tf.keras.models.load_model(ARTIFACTS_DIR / "best_model_unified.keras") # Load Keras model
    else: # If sklearn
        model = load(ARTIFACTS_DIR / "best_model_unified.joblib") # Load sklearn model

except (AttributeError, KeyError, Exception) as e: # Catch loading errors
    print(f"ERROR: Compatibility issue or missing files: {e}") # Print error
    print(">>> FIX: Please run 'Module 01' and 'Module 02' again.") # Print fix
    sys.exit(1) # Exit the script

X_test = np.load(ARTIFACTS_DIR / "X_test.npz")["X"] # Load preprocessed testing features
y_test = np.load(ARTIFACTS_DIR / "y_test.npy") # Load testing labels

print(f"Artifacts loaded successfully. Active Model Type: {model_type}") # Print confirmation

# ===================== # Header Section
# FEATURE NAMES         # Feature Names Header
# ===================== # Header Section

feature_names = preprocessor.get_feature_names_out() # Get feature names from preprocessor

# ===================== # Header Section
# GLOBAL INTERPRETABILITY # Global Interpretability Header
# ===================== # Header Section

if hasattr(model, "coef_"): # Check if the model is a linear model
    # Logistic Regression case
    coef_df = pd.DataFrame({ # Create a DataFrame for model coefficients
        "Feature": feature_names, # Assign feature names
        "Coefficient": model.coef_[0] # Assign corresponding coefficient values
    }) # End of DataFrame init
    coef_df["AbsCoeff"] = coef_df["Coefficient"].abs() # Calculate absolute values
    coef_df = coef_df.sort_values("AbsCoeff", ascending=False) # Sort by importance

    sns.barplot(x="Coefficient", y="Feature", data=coef_df.head(15)) # Create bar plot
    plt.title("Top 15 Features – Linear Model Importance") # Set title
    plt.show() # Display

elif hasattr(model, "feature_importances_"): # Check if model is tree-based
    # Random Forest case
    imp_df = pd.DataFrame({ # Create a DataFrame for feature importances
        "Feature": feature_names, # Assign feature names
        "Importance": model.feature_importances_ # Assign importance scores
    }).sort_values("Importance", ascending=False) # Sort features

    sns.barplot(x="Importance", y="Feature", data=imp_df.head(15)) # Create bar plot
    plt.title("Top 15 Features – Tree Model Importance") # Set title
    plt.show() # Display

# ===================== # Header Section
# SHAP EXPLAINABILITY   # SHAP Explainability Header
# ===================== # Header Section

# Use a robust way to initialize the SHAP explainer                     # Unified Explainer Section
if model_type == "keras": # If Deep Learning
    # Use KernelExplainer for Keras models (agnostic and stable)
    # Using a small background set for speed
    explainer = shap.KernelExplainer(model.predict, X_test[:20], verbose=0) # Init explainer
    shap_values = explainer.shap_values(X_test[:10]) # Calculate for first 10 samples
    shap_vals_raw = shap_values[0] if isinstance(shap_values, list) else shap_values # Handle output format
else: # If Classical ML
    if hasattr(model, "feature_importances_"): # If tree-based
        explainer = shap.TreeExplainer(model) # Use TreeExplainer
        shap_values = explainer(X_test) # Calculate
    else: # If other (Linear, KNN, etc.)
        explainer = shap.Explainer(model.predict, X_test) # Use general Explainer
        shap_values = explainer(X_test) # Calculate

    # Normalize output format for plots                                 # Format normalization
    shap_vals_raw = shap_values.values if hasattr(shap_values, "values") else shap_values
    if len(shap_vals_raw.shape) == 3: shap_vals_raw = shap_vals_raw[:, :, 1] # Select positive class

# ===================== # Header Section
# SHAP SUMMARY          # SHAP Summary Header
# ===================== # Header Section

print("\nGenerating SHAP plots...") # Status
shap.summary_plot( # Create a SHAP summary dot plot
    shap_vals_raw, # Use processed values
    X_test[:10] if model_type == "keras" else X_test, # Match sample size
    feature_names=feature_names # Label features
) # End summary

# ===================== # Header Section
# LOCAL EXPLANATION     # Local Explanation Header
# ===================== # Header Section

patient_idx = 0 # Select patient index
print(f"\nExplaining prediction for Patient #{patient_idx}...") # Status

if hasattr(shap_values, "values") and model_type != "keras": # Waterfall works on Explanation objects
    shap.plots.waterfall(shap_values[patient_idx], max_display=15) # Show waterfall
else: # Fallback to bar plot for manual values
    shap.bar_plot(shap_vals_raw[patient_idx], feature_names=feature_names, max_display=15) # Show bar
