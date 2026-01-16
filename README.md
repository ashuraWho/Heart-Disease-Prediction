# Heart Disease Prediction Project

This project implements a comprehensive machine learning pipeline to predict the presence of heart disease based on clinical data. It covers everything from exploratory data analysis (EDA) and preprocessing to classical machine learning models, model explainability, and deep learning.

## Project Structure

The project is organized into sequential modules located in the `notebooks/` directory:

1.  **`01_EDA_Preprocessing.py`**:
    - Performs exploratory data analysis.
    - Handles data cleaning and feature renaming.
    - Implements a preprocessing pipeline (scaling for numerical features, one-hot encoding for categorical features).
    - Saves processed data and the fitted preprocessor as artifacts.

2.  **`02_ML_Classic.py`**:
    - Loads preprocessed artifacts.
    - Trains and tunes several classical models (Logistic Regression, KNN, SVM, Random Forest) using Grid Search and Cross-Validation.
    - Evaluates models based on Recall, Accuracy, Precision, F1, and ROC-AUC.
    - Saves the best-performing classical model.

3.  **`03_Explainability.py`**:
    - Focuses on model interpretability.
    - Uses SHAP (SHapley Additive exPlanations) to explain model predictions.
    - Visualizes global feature importance and local explanations for individual patients.

4.  **`04_Deep_Learning.py`**:
    - Implements a Multi-Layer Perceptron (MLP) using TensorFlow/Keras.
    - Employs robust regularization techniques like L2 regularization, Batch Normalization, and Dropout to handle the small tabular dataset.
    - Compares deep learning performance with classical approaches.

## Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib shap tensorflow
```

### Usage

The modules are designed to be run in order. From the project root, execute:

```bash
python notebooks/01_EDA_Preprocessing.py
python notebooks/02_ML_Classic.py
python notebooks/03_Explainability.py
python notebooks/04_Deep_Learning.py
```

## Dataset

The dataset used is `Heart_Disease_Prediction.csv`, located in the `data/` directory. It contains clinical parameters such as age, blood pressure, cholesterol levels, and more, with the target variable being the presence or absence of heart disease.

## Key Features

- **Robust Path Handling**: Uses `pathlib` for cross-platform compatibility.
- **Reproducibility**: Global seeds are set for consistent results.
- **Explainability**: Integrated SHAP analysis for transparent AI.
- **Line-by-Line Documentation**: Every module is extensively commented for educational clarity.

## Results

Model performance is evaluated with a focus on **Recall**, which is critical in medical diagnostics to minimize false negatives. Detailed metrics and plots (ROC curves, confusion matrices) are generated for each approach.

## Troubleshooting (macOS / Anaconda)

If you encounter a **segmentation fault** or crashes when running the deep learning module on macOS (especially with Anaconda base environment), please follow these steps:

1.  **Use the Setup Script**: We have provided a `setup_mac.sh` script to automate the creation of a stable environment. Run:
    ```bash
    chmod +x setup_mac.sh
    ./setup_mac.sh
    ```
    After it finishes, activate the environment: `conda activate heart_disease`.

2.  **Environment Variables**: The scripts already include several fixes:
    - `KMP_DUPLICATE_LIB_OK=True`: Fixes multiple OpenMP runtime conflicts.
    - `TF_ENABLE_ONEDNN_OPTS=0`: Disables unstable CPU optimizations.
    - `OMP_NUM_THREADS=1`: Prevents resource-related SegFaults.
    - `CUDA_VISIBLE_DEVICES=-1`: Forces CPU execution to bypass unstable GPU drivers on Mac.

3.  **Avoid Base Environment**: **Do not run the code in Anaconda's (base) environment.** The base environment is prone to binary conflicts between MKL-linked libraries and pip-installed TensorFlow.

4.  **Apple Silicon (M1/M2/M3/M4)**: For Mac with Apple Silicon, ensure you are in an ARM64 terminal and run:
    ```bash
    pip install tensorflow-macos tensorflow-metal
    ```
