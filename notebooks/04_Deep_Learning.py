# ============================================================ # Module 04 – Deep Learning (MLP)
# NOTEBOOK 4 – DEEP LEARNING (MULTI-LAYER PERCEPTRON)          # Heart Disease Prediction
# Heart Disease Prediction                                     # Global Header
# ============================================================ # Global Header
#
# OBJECTIVES:                                                  # List of objectives
# - Evaluate an MLP on a small tabular dataset                  # Objective 1
# - Apply robust regularization techniques                      # Objective 2
# - Conceptual comparison between DL and classical ML           # Objective 3
#
# ARCHITECTURAL NOTE:                                          # Note section
# - This script loads preprocessed data from /artifacts        # Note item 1
# - It is designed to be run after Notebook 01                 # Note item 2
# ============================================================ # Footer for header

# ===================== # Header Section
# 1. IMPORT LIBRARIES   # Import Libraries Header
# ===================== # Header Section

import os # Import os for environment variable manipulation
import sys # Import sys to check the Python environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix for common segmentation fault on macOS/Anaconda (multiple OpenMP runtimes)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN to improve stability on some CPUs
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads to 1 to prevent resource-related SegFaults
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU execution to bypass potentially unstable GPU/Metal drivers on Mac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TensorFlow logging noise to focus on errors

# --- DIAGNOSTIC: Check Environment ---
print(f"Python Executable: {sys.executable}") # Print the path to the current Python interpreter
print(f"Python Version: {sys.version}") # Print the version of Python being used
if "anaconda3/bin/python" in sys.executable or "miniconda3/bin/python" in sys.executable: # Check if running in base
    print("WARNING: You are likely running in the 'base' environment. This is discouraged.") # Warn the user
# -------------------------------------

print("Importing core libraries...") # Diagnostic print
import random # Import random for reproducibility
import numpy as np # Import numpy for numerical operations
print("Importing TensorFlow... (Potential crash point)") # Diagnostic print
import tensorflow as tf # Import tensorflow for deep learning
print("TensorFlow imported successfully.") # Diagnostic print

import pandas as pd # Import pandas for data handling

import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for visualization

from tensorflow.keras.models import Sequential # Import Sequential model from Keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input # Import layers for building MLP
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping callback
from tensorflow.keras.regularizers import l2 # Import L2 regularization

from sklearn.metrics import ( # Import metrics from sklearn
    accuracy_score, # Import Accuracy metric
    precision_score, # Import Precision metric
    recall_score, # Import Recall metric
    f1_score, # Import F1 metric
    roc_auc_score, # Import ROC-AUC metric
    confusion_matrix, # Import Confusion Matrix
    RocCurveDisplay # Import ROC Curve Display
) # End of metrics import

sns.set(style="whitegrid") # Set whitegrid style for seaborn plots
plt.rcParams["figure.figsize"] = (10, 6) # Set default figure size

# ===================== # Header Section
# 2. REPRODUCIBILITY    # Reproducibility Header
# ===================== # Header Section

SEED = 42 # Define a global seed for reproducibility
random.seed(SEED) # Set seed for Python's built-in random module
np.random.seed(SEED) # Set seed for numpy's random number generator
tf.random.set_seed(SEED) # Set seed for TensorFlow

# ===================== # Header Section
# 3. PATH HANDLING & LOAD DATA # Path and Load Data Header
# ===================== # Header Section

from pathlib import Path # Import Path for robust filesystem path manipulation
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Determine the project root directory
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" # Define the artifacts directory path

X_train = np.load(ARTIFACTS_DIR / "X_train.npz")["X"] # Load training features from artifacts
X_test  = np.load(ARTIFACTS_DIR / "X_test.npz")["X"] # Load testing features from artifacts

y_train = np.load(ARTIFACTS_DIR / "y_train.npy") # Load training labels from artifacts
y_test  = np.load(ARTIFACTS_DIR / "y_test.npy") # Load testing labels from artifacts

input_dim = X_train.shape[1] # Get the number of input dimensions (features)

# ===================== # Header Section
# 4. MODEL ARCHITECTURE # Model Architecture Header
# ===================== # Header Section
# Compact MLP designed for small tabular data                  # Architecture comment

model = Sequential([ # Initialize a Sequential Keras model
    Input(shape=(input_dim,)), # Define the input layer with the correct shape (recommended in Keras 3)
    
    Dense( # Define the first hidden layer
        32, # Number of neurons
        activation="relu", # Use ReLU activation function
        kernel_regularizer=l2(1e-3) # Apply L2 regularization to prevent overfitting
    ), # End of layer
    BatchNormalization(), # Apply Batch Normalization for training stability
    Dropout(0.4), # Apply Dropout for regularization

    Dense( # Define the second hidden layer
        16, # Number of neurons
        activation="relu", # Use ReLU activation function
        kernel_regularizer=l2(1e-3) # Apply L2 regularization
    ), # End of layer
    BatchNormalization(), # Apply Batch Normalization
    Dropout(0.3), # Apply Dropout

    Dense( # Define the third hidden layer
        8, # Number of neurons
        activation="relu", # Use ReLU activation function
        kernel_regularizer=l2(1e-3) # Apply L2 regularization
    ), # End of layer

    Dense(1, activation="sigmoid") # Define the output layer with sigmoid for binary classification
]) # End of Sequential model

# ===================== # Header Section
# 5. COMPILATION        # Compilation Header
# ===================== # Header Section

model.compile( # Compile the model
    optimizer=Adam(learning_rate=1e-3), # Use Adam optimizer with a specific learning rate
    loss="binary_crossentropy", # Use binary crossentropy loss for binary classification
    metrics=["accuracy"] # Track accuracy during training
) # End of compile

model.summary() # Print the model summary showing layers and parameters

# ===================== # Header Section
# 6. EARLY STOPPING     # Early Stopping Header
# ===================== # Header Section

early_stopping = EarlyStopping( # Define EarlyStopping callback
    monitor="val_loss", # Monitor validation loss
    patience=20, # Wait for 20 epochs without improvement before stopping
    restore_best_weights=True # Restore model weights from the best epoch
) # End of EarlyStopping init

# ===================== # Header Section
# 7. TRAINING           # Training Header
# ===================== # Header Section

print("Starting model training... (Potential crash point on Mac)") # Diagnostic print
history = model.fit( # Train the model
    X_train, # Training features
    y_train, # Training labels
    validation_split=0.2, # Use 20% of training data for validation
    epochs=300, # Maximum number of epochs
    batch_size=16, # Batch size
    callbacks=[early_stopping], # Use early stopping callback
    verbose=1 # Show training progress
) # End of fit

# ===================== # Header Section
# 8. TRAINING DIAGNOSTICS # Diagnostics Header
# ===================== # Header Section

history_df = pd.DataFrame(history.history) # Convert training history to a DataFrame

# Loss plot                                                     # Loss Plot Comment
plt.plot(history_df["loss"], label="Train") # Plot training loss
plt.plot(history_df["val_loss"], label="Validation") # Plot validation loss
plt.title("Training Loss") # Set plot title
plt.xlabel("Epoch") # Set X-axis label
plt.ylabel("Loss") # Set Y-axis label
plt.legend() # Show legend
plt.show() # Display the plot

# Accuracy plot                                                 # Accuracy Plot Comment
plt.plot(history_df["accuracy"], label="Train") # Plot training accuracy
plt.plot(history_df["val_accuracy"], label="Validation") # Plot validation accuracy
plt.title("Training Accuracy") # Set plot title
plt.xlabel("Epoch") # Set X-axis label
plt.ylabel("Accuracy") # Set Y-axis label
plt.legend() # Show legend
plt.show() # Display the plot

# ===================== # Header Section
# 9. TEST SET EVALUATION # Evaluation Header
# ===================== # Header Section

y_pred_proba = model.predict(X_test).ravel() # Predict probabilities on test set and flatten the result
y_pred = (y_pred_proba >= 0.5).astype(int) # Convert probabilities to binary predictions using 0.5 threshold

nn_results = { # Store performance metrics in a dictionary
    "Accuracy": accuracy_score(y_test, y_pred), # Calculate accuracy
    "Precision": precision_score(y_test, y_pred), # Calculate precision
    "Recall": recall_score(y_test, y_pred), # Calculate recall
    "F1-score": f1_score(y_test, y_pred), # Calculate F1-score
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba) # Calculate ROC-AUC
} # End of dictionary

print("\nNeural Network Performance:") # Print header for NN performance
for k, v in nn_results.items(): # Iterate through metrics
    print(f"{k}: {v:.3f}") # Print each metric formatted to 3 decimal places

# ===================== # Header Section
# 10. ROC CURVE         # ROC Curve Header
# ===================== # Header Section

RocCurveDisplay.from_predictions( # Create ROC curve display from predictions
    y_test, # Ground truth labels
    y_pred_proba, # Predicted probabilities
    name="Neural Network" # Name for the plot legend
) # End of from_predictions

plt.title("ROC Curve – Neural Network") # Set plot title
plt.show() # Display the plot

# ===================== # Header Section
# 11. CONFUSION MATRIX  # Confusion Matrix Header
# ===================== # Header Section

cm = confusion_matrix(y_test, y_pred) # Calculate the confusion matrix

sns.heatmap( # Create a heatmap for the confusion matrix
    cm, # Confusion matrix data
    annot=True, # Annotate cells with values
    fmt="d", # Use integer formatting for annotations
    cmap="Blues" # Use Blues color map
) # End of heatmap

plt.title("Confusion Matrix – Neural Network") # Set plot title
plt.xlabel("Predicted") # Set label for X-axis
plt.ylabel("Actual") # Set label for Y-axis
plt.show() # Display the plot

# ===================== # Header Section
# 12. SAVE MODEL        # Save Model Header
# ===================== # Header Section

model.save(ARTIFACTS_DIR / "nn_model.keras") # Save the trained Keras model to artifacts

# ===================== # Header Section
# 13. CONCLUSIONS       # Conclusions Header
# ===================== # Header Section
# - On small tabular datasets, DL is often not the best solution # Conclusion 1
# - Classical ML often:                                         # Classical ML advantages
#   • Generalizes better                                        # Advantage 1
#   • Is more interpretable                                     # Advantage 2
#   • Is less computationally expensive                         # Advantage 3
#
# This script demonstrates methodological awareness             # Final Note 1
# rather than naive use of Deep Learning.                       # Final Note 2
