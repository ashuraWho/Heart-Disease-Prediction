# Heart Disease Prediction: A Robust and Explainable ML Pipeline

## 🌟 Project Vision & Mission

**What we are doing:**
This project is an end-to-end Machine Learning ecosystem designed to predict the presence of heart disease from clinical data. We have built a robust, portable, and highly documented pipeline that transforms raw clinical observations into actionable insights and predictive models.

**Where we are going:**
The goal is to bridge the gap between "Black Box" algorithms and clinical interpretability. By combining classical statistical models with modern Deep Learning and cutting-edge Explainable AI (XAI) techniques like SHAP, we are moving toward a future where AI-assisted diagnosis is not only accurate but also transparent and justifiable to medical professionals.

---

## 🛠 Architectural Decisions & Motivations

Every line of code and every structural choice in this repository was made with specific technical and educational goals in mind.

### 1. Robust Path Management (`pathlib`)
*   **Motivation:** Traditional string-based path handling (e.g., `../data/file.csv`) is brittle and often fails depending on where the script is executed.
*   **Solution:** We implemented `pathlib.Path(__file__).resolve()`. This ensures the project root is always identified correctly relative to the script's location, making the entire pipeline portable across Windows, macOS, and Linux.

### 2. Environment Diagnostics & Anaconda Stability
*   **Motivation:** macOS users often experience "Segmentation Faults" when running TensorFlow within Anaconda, especially when scripts inadvertently use the Anaconda 'base' environment instead of a dedicated one.
*   **Solution:** We added diagnostic prints to every module that output the exact `sys.executable` being used. If the script detects it is running in the `anaconda3/bin/python` (base) path, it issues a critical warning.

### 3. macOS-Specific Stability Fixes
*   **KMP_DUPLICATE_LIB_OK=True:** Resolves conflicts when multiple OpenMP runtimes (common in Anaconda/Intel libraries) are loaded.
*   **OMP_NUM_THREADS=1:** Prevents deadlocks and resource-related crashes by limiting threading during initialization.
*   **CUDA_VISIBLE_DEVICES=-1:** Forces CPU execution on Mac. Given the small size of tabular medical data, CPU training is nearly as fast as GPU/Metal but significantly more stable across different macOS versions.
*   **TF_ENABLE_ONEDNN_OPTS=0:** Disables floating-point optimizations that are known to cause arithmetic errors or crashes on certain CPU architectures.

### 4. Robust Explainable AI (SHAP)
*   **Motivation:** SHAP often fails or produces different output shapes depending on the model (Tree-based vs. Linear vs. KNN).
*   **Solution:** We built a custom "SHAP Wrapper" logic in Module 03. It automatically detects the model type, chooses the correct Explainer, and slices the output dimensions to ensure it always correctly identifies the "Presence" of heart disease (index 1) regardless of the internal model representation.

### 5. Transition to Keras 3
*   **Motivation:** Older Keras code is becoming deprecated.
*   **Solution:** Module 04 follows Keras 3 standards, using explicit `Input()` layers. This ensures the model is future-proof and compatible with the latest TensorFlow releases.

### 6. Line-by-Line English Documentation
*   **Motivation:** To provide maximum educational value and ensure the logic is clear to developers of all levels.
*   **Solution:** Every single line of code in the `notebooks/` directory is followed by a `#` comment in English, explaining its purpose and context.

---

## 📂 Pipeline Deep-Dive

### Module 01: EDA & Preprocessing
The foundation of the project. It focuses on "Data Quality First."
- **Motivations:** Raw data is messy. We implement categorical encoding and numerical scaling within a `ColumnTransformer` to ensure zero data leakage between training and testing sets.
- **Output:** Processed `.npz` and `.npy` artifacts for subsequent modules.

### Module 02: Classical Machine Learning
Baselining and model selection.
- **Motivations:** We don't just pick a model; we compete them. Logistic Regression, KNN, SVM, and Random Forest are tuned via `GridSearchCV`.
- **Focus:** We prioritize **Recall**. In heart disease prediction, a "False Negative" (missing a sick patient) is far more dangerous than a "False Positive."

### Module 03: Explainability (SHAP)
Opening the "Black Box."
- **Motivations:** Doctors don't trust a number; they trust a reason. This module uses SHAP values to show which features (e.g., Cholesterol, Chest Pain type) most influenced the model's decision for a specific patient.

### Module 04: Deep Learning (MLP)
Advanced predictive modeling.
- **Motivations:** For complex patterns, we use a Multi-Layer Perceptron.
- **Robustness:** We use heavy regularization (L2, Dropout, Batch Normalization) to prevent the neural network from overfitting on this relatively small tabular dataset.

### Module 05: Real-World Inference
The final utility of the project.
- **What it does:** This module loads the best-performing model and the preprocessor to predict the risk for a **new patient** based on clinical input.
- **Value:** It demonstrates the transition from a training pipeline to a functional medical decision-support tool.

---

## 🚀 Getting Started

### 1. The Right Way (macOS / Linux)
We provide a dedicated setup script to bypass common environment issues:
```bash
chmod +x setup_mac.sh
./setup_mac.sh
conda activate heart_disease
python -m pip install -r requirements.txt
```

### 2. Manual Installation
```bash
pip install -r requirements.txt
```

### 3. Execution Order
Always run the modules in sequence:
1. `python notebooks/01_EDA_Preprocessing.py`
2. `python notebooks/02_ML_Classic.py`
3. `python notebooks/03_Explainability.py`
4. `python notebooks/04_Deep_Learning.py`
5. `python notebooks/05_Inference.py`

---

## 🏥 How to Predict Heart Disease for a New Patient

Once you have run the training modules (01 and 02), you can use `notebooks/05_Inference.py` to make predictions on new data.

### Steps to predict:
1.  Open `notebooks/05_Inference.py`.
2.  Locate the `new_patient_data` dictionary.
3.  Modify the clinical values (Age, Cholesterol, etc.) to match your patient's data.
4.  Run the script:
    ```bash
    python notebooks/05_Inference.py
    ```
5.  The script will output the probability of heart disease presence and the final diagnostic recommendation.

---

## 🛑 Critical Troubleshooting

### Segmentation Fault on Mac?
**NEVER** run the scripts like this: `/opt/anaconda3/bin/python notebooks/04_Deep_Learning.py`.
**ALWAYS** run them like this:
1. `conda activate heart_disease`
2. `python notebooks/04_Deep_Learning.py`

Running with an absolute path to the Anaconda base Python will bypass your environment and trigger a crash.

---

## 🗺 Roadmap
- [ ] Integration of SMOTE for better class balancing.
- [ ] Deployment of the pipeline as a REST API (FastAPI).
- [ ] Frontend dashboard for real-time patient risk assessment.
- [ ] Cross-validation for the Deep Learning module.
