# ============================================================ # Module 03 – Explainability & Interpretability
# MODULE 03 – EXPLAINABILITY & INTERPRETABILITY               # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header

import os # Import os for environment variable manipulation
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda

from pathlib import Path # Import Path for robust filesystem path manipulation
import numpy as np # Import numpy for numerical operations and array handling
import pandas as pd # Import pandas for data manipulation and tabular data

import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for statistical data visualization
sns.set(style="whitegrid") # Set a clean, whitegrid plotting style for seaborn
plt.rcParams["figure.figsize"] = (10, 6) # Set default figure size for consistency

import shap # Import shap for model explainability
from joblib import load # Import load from joblib to retrieve saved models

# ===================== # Header Section
# PATHS                 # Paths Header
# ===================== # Header Section

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Define the root directory of the project
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

# ===================== # Header Section
# LOAD ARTIFACTS        # Load Artifacts Header
# ===================== # Header Section

preprocessor = load(ARTIFACTS_DIR / "preprocessor.joblib") # Load the saved preprocessing pipeline
model = load(ARTIFACTS_DIR / "best_model_classic.joblib") # Load the best classical model

X_test = np.load(ARTIFACTS_DIR / "X_test.npz")["X"] # Load preprocessed testing features
y_test = np.load(ARTIFACTS_DIR / "y_test.npy") # Load testing labels

print("Artifacts loaded successfully") # Print confirmation message for artifact loading

# ===================== # Header Section
# FEATURE NAMES         # Feature Names Header
# ===================== # Header Section

num_features = preprocessor.transformers_[0][2] # Extract numerical feature names from the preprocessor

cat_features = ( # Extract categorical feature names after one-hot encoding
    preprocessor # Reference the main preprocessor
    .transformers_[1][1] # Get the categorical transformer pipeline
    .named_steps["onehot"] # Access the onehot step
    .get_feature_names_out( # Generate feature names based on original categories
        preprocessor.transformers_[1][2] # Use the list of categorical feature names as input
    ) # End of name generation
) # End of cat_features assignment

feature_names = np.concatenate([num_features, cat_features]) # Concatenate numerical and categorical feature names

# ===================== # Header Section
# GLOBAL INTERPRETABILITY # Global Interpretability Header
# ===================== # Header Section

if hasattr(model, "coef_"): # Check if the model is a linear model (e.g., Logistic Regression)
    # Logistic Regression case
    coef_df = pd.DataFrame({ # Create a DataFrame for model coefficients
        "Feature": feature_names, # Assign feature names
        "Coefficient": model.coef_[0] # Assign corresponding coefficient values
    }) # End of DataFrame init
    coef_df["AbsCoeff"] = coef_df["Coefficient"].abs() # Calculate absolute coefficient values for importance ranking
    coef_df = coef_df.sort_values("AbsCoeff", ascending=False) # Sort features by absolute importance

    sns.barplot( # Create a bar plot for the top coefficients
        x="Coefficient", # Coefficients on the X-axis
        y="Feature", # Feature names on the Y-axis
        data=coef_df.head(15) # Use only the top 15 features
    ) # End of barplot
    plt.title("Top 15 Features – Logistic Regression") # Set title for coefficient plot
    plt.show() # Display the coefficient plot

elif hasattr(model, "feature_importances_"): # Check if the model has feature importances (e.g., Random Forest)
    # Random Forest case
    imp_df = pd.DataFrame({ # Create a DataFrame for feature importances
        "Feature": feature_names, # Assign feature names
        "Importance": model.feature_importances_ # Assign importance scores
    }).sort_values("Importance", ascending=False) # Sort features by importance score

    sns.barplot( # Create a bar plot for feature importances
        x="Importance", # Importance scores on the X-axis
        y="Feature", # Feature names on the Y-axis
        data=imp_df.head(15) # Use only the top 15 features
    ) # End of barplot
    plt.title("Top 15 Features – Random Forest") # Set title for importance plot
    plt.show() # Display the importance plot

# ===================== # Header Section
# SHAP EXPLAINABILITY   # SHAP Explainability Header
# ===================== # Header Section

# Use a robust way to initialize the SHAP explainer
if hasattr(model, "feature_importances_"): # If the model is tree-based (like Random Forest)
    explainer = shap.TreeExplainer(model) # Use TreeExplainer for optimized tree analysis
    shap_values = explainer(X_test) # Calculate SHAP values
elif hasattr(model, "coef_"): # If the model is linear (like Logistic Regression)
    explainer = shap.LinearExplainer(model, X_test) # Use LinearExplainer for linear models
    shap_values = explainer(X_test) # Calculate SHAP values
else: # For other models (like KNN or SVM)
    # Use KernelExplainer as a model-agnostic approach
    # We use a small subset of test data as a background for speed if needed, but here we use X_test
    # We wrap the prediction function to ensure SHAP can call it
    if hasattr(model, "predict_proba"): # Prefer probability predictions if available
        # Wrap predict_proba to return only the positive class probability if necessary,
        # or handle the multi-class output of SHAP
        explainer = shap.Explainer(model.predict_proba, X_test) # Initialize general explainer
    else: # Fallback to hard predictions
        explainer = shap.Explainer(model.predict, X_test) # Initialize general explainer
    shap_values = explainer(X_test) # Calculate SHAP values

# Handle SHAP output dimensions (multi-class vs binary)
# If SHAP values have an extra dimension for classes, select the positive class (index 1)
try:
    if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3: # Check if output is (samples, features, classes)
        shap_obj = shap_values[:, :, 1] # Select SHAP values for the 'Presence' class
    elif not hasattr(shap_values, "values") and len(shap_values.shape) == 3: # Case for raw numpy array
        shap_obj = shap_values[:, :, 1] # Select positive class
    else: # If output is already (samples, features)
        shap_obj = shap_values # Use as is
except Exception: # Fallback
    shap_obj = shap_values # Default to original object

# Extract raw values for summary plots if it's an Explanation object
shap_vals_raw = shap_obj.values if hasattr(shap_obj, "values") else shap_obj

# ===================== # Header Section
# SHAP SUMMARY          # SHAP Summary Header
# ===================== # Header Section

shap.summary_plot( # Create a SHAP summary dot plot
    shap_vals_raw, # Use raw SHAP values
    X_test, # Use corresponding feature values
    feature_names=feature_names # Provide feature names for labeling
) # End of summary_plot

shap.summary_plot( # Create a SHAP summary bar plot (global importance)
    shap_vals_raw, # Use raw SHAP values
    X_test, # Use corresponding feature values
    feature_names=feature_names, # Provide feature names
    plot_type="bar" # Specify bar plot type
) # End of summary_plot

# ===================== # Header Section
# LOCAL EXPLANATION     # Local Explanation Header
# ===================== # Header Section

patient_idx = 0 # Select an index for a specific patient to explain

shap.plots.waterfall( # Create a waterfall plot for local explanation
    shap_obj[patient_idx], # Provide SHAP values for the specific patient
    max_display=15 # Limit the number of displayed features
) # End of waterfall plot
