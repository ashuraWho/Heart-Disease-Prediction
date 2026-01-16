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

If you encounter a **segmentation fault** or crashes when running the deep learning module on macOS (especially with Anaconda), try the following:

1.  **Environment Variables**: The scripts already include `KMP_DUPLICATE_LIB_OK=True`, `TF_ENABLE_ONEDNN_OPTS=0`, and `OMP_NUM_THREADS=1` to improve stability and prevent OpenMP conflicts.
2.  **Force CPU**: If the crash persists, you can force TensorFlow to use the CPU by adding `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` at the top of `04_Deep_Learning.py`.
3.  **Clean Environment**: Anaconda's `numpy` (which uses MKL) often conflicts with `tensorflow`. It is highly recommended to:
    -   Create a new conda environment: `conda create -n heart_disease python=3.10`
    -   Activate it: `conda activate heart_disease`
    -   Install dependencies ONLY via pip: `pip install -r requirements.txt`
4.  **Apple Silicon (M1/M2/M3/M4)**: If you are on an ARM-based Mac, ensure you are using an ARM64 version of Python. For optimized performance, consider installing:
    ```bash
    pip install tensorflow-macos tensorflow-metal
    ```
