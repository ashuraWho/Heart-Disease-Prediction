# Heart Disease Prediction: A Robust and Explainable ML Pipeline (V3)

## 🌟 Project Vision & Mission

**What we are doing:**
This project is an end-to-end Machine Learning ecosystem designed to predict heart disease using a comprehensive 21-parameter clinical schema. We have unified classical statistical models and Deep Learning into a single competitive training framework to ensure we always use the most accurate mathematical approach for your specific data environment.

**Where we are going:**
We aim to provide a reliable, persistent Clinical Decision Support System. By integrating SQL storage, automated model competition, and Explainable AI (SHAP), we provide clinicians with a tool that doesn't just give a "Yes/No" but also provides a "Why" backed by historical patient data.

---

## 🏥 Clinical Dataset Structure (21 Columns)

The system is optimized for:
- **Demographics:** Age, Gender.
- **Vitals:** Blood Pressure, BMI, Sleep Hours.
- **Biochemistry:** Cholesterol, Triglycerides, Fasting Sugar, CRP, Homocysteine.
- **Lifestyle:** Exercise, Smoking, Alcohol, Stress, Sugar.
- **History:** Family History, Diabetes, Hypertension, HDL/LDL levels.
- **Target:** Heart Disease Status (Yes/No).

---

## 🛠 Architectural Decisions & Motivations

### 1. Unified Model Competition (ML vs DL)
- **Motivation:** Deep Learning is not always better for small tabular datasets.
- **Solution:** Module 02 now automatically trains multiple classical models (LR, SVM, RF) and a Neural Network. It evaluates all using the **F1-Score** (to balance False Positives and False Negatives) and automatically saves the "Winner" as the active production model.

### 2. Explicit Feature Mapping
- **Motivation:** String labels prevent mathematical analysis (correlations).
- **Solution:** Module 01 now maps all text (e.g., "High", "Male", "Yes") into standardized numeric scales. This reduces model bias and enables the generation of high-quality correlation heatmaps.

### 3. Persistent SQL Registry
- **Motivation:** Historical clinical records are vital for patient tracking.
- **Solution:** Integrated SQLite (`patients_data.db`). All predictions are logged, allowing for batch re-runs if the underlying models are updated.

---

## 📂 Pipeline Modules

- **Module 01: EDA & Preprocessing:** Data cleaning, explicit mapping, and interactive plotting.
- **Module 02: Unified Training:** The competition between ML and DL models.
- **Module 03: Explainability (XAI):** SHAP analysis explaining the "Winner's" decisions.
- **Module 04: Real-World Inference:** The interactive clinic interface and SQL logger.
- **Main Dashboard (`main.py`):** The central orchestration menu.

---

## 🚀 Getting Started

### 1. Integrated Dashboard
```bash
python main.py
```

### 2. Maintenance
- **System Reset:** If you change your Python environment, use the 'r' option in the menu to clear artifacts and regenerate models locally.
- **Database Clear:** Use the 'd' option to wipe all stored patient history.

---

## 🛑 Technical Stability (macOS / Anaconda)
The system is pre-configured with `KMP_DUPLICATE_LIB_OK=True` and `OMP_NUM_THREADS=1` to prevent crashes common on Mac machines. It also defaults to **CPU execution** for maximum stability.

---

## 🗺 Roadmap
- [ ] Integration of Synthetic Minority Over-sampling Technique (SMOTE).
- [ ] PDF Report Generation for patient consults.
- [ ] Integration with EHR (Electronic Health Record) standards.
