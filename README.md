# Heart Disease Prediction: A Robust and Explainable ML Pipeline (V2)

## 🌟 Project Vision & Mission

**What we are doing:**
This project is an end-to-end Machine Learning ecosystem designed to predict the presence of heart disease using an expanded clinical dataset of 21 parameters. We have built a robust, persistent, and highly documented pipeline that transforms complex medical observations into Actionable Clinical Intelligence.

**Where we are going:**
We aim to empower medical professionals with a guided Decision Support System. By combining persistent SQL storage, interactive data entry, and Explainable AI (SHAP), we provide a tool that not only predicts risk but also explains the underlying clinical factors, fostering trust between the algorithm and the clinician.

---

## 🏥 Clinical Dataset Structure (21 Columns)

The system is now optimized for an expanded dataset including:
- **Demographics:** Age, Gender.
- **Vitals:** Blood Pressure (Systolic), BMI, Sleep Hours.
- **Biochemistry:** Cholesterol Level, Triglyceride Level, Fasting Blood Sugar, CRP Level (Inflammation), Homocysteine Level.
- **Lifestyle:** Exercise Habits, Smoking, Alcohol Consumption, Stress Level, Sugar Consumption.
- **History:** Family Heart Disease, Diabetes, Pre-existing High BP, Low HDL, High LDL.
- **Target:** Heart Disease Status (Yes/No).

---

## 🛠 Architectural Decisions & Motivations

### 1. SQL Persistence
- **Motivation:** Patient data should not be lost between sessions.
- **Solution:** Integrated an SQLite database (`patients_data.db`). Every interactive prediction is automatically logged with a timestamp, allowing for historical review and batch re-processing.

### 2. Interactive Decision Support
- **Motivation:** Clinicians need a guided experience.
- **Solution:** Option 5 in the dashboard provides a step-by-step input flow with integrated clinical definitions for every parameter.

### 3. Data Integrity (Missing Values)
- **Motivation:** Real-world medical data is often incomplete.
- **Solution:** Module 01 now strictly enforces `df.dropna()` to ensure the model is trained only on high-quality, complete clinical records.

---

## 📂 Pipeline Modules

- **Module 01: EDA & Preprocessing:** Handles the new 21-column schema and removes missing values.
- **Module 02: Classical ML:** Trains and tunes Logistic Regression, KNN, SVM, and Random Forest.
- **Module 03: Explainability:** Uses SHAP to visualize the weight of clinical factors.
- **Module 04: Deep Learning:** Multi-Layer Perceptron with robust regularization.
- **Module 05: Real-World Inference:** The interactive interface for patient data entry and SQL storage.
- **Main Dashboard (`main.py`):** The central command center with match-case navigation.

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
chmod +x setup_mac.sh
./setup_mac.sh
conda activate heart_disease
python -m pip install -r requirements.txt
```

### 2. Integrated Dashboard
Launch the unified interface:
```bash
python main.py
```

---

## 🛑 Maintenance & Troubleshooting

### Reset Options
- **Option 'd':** Deletes the SQL database if you want to clear patient history.
- **Option 'r':** Resets all ML artifacts (models/preprocessors). Use this if you change your environment or see an `AttributeError`.

### macOS SegFaults
The system includes `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS=1`, and `CUDA_VISIBLE_DEVICES=-1` to ensure maximum stability on Mac/Anaconda.

---

## 🗺 Roadmap
- [ ] Integration of SMOTE for class balancing.
- [ ] Automated PDF Report Generation for patients.
- [ ] REST API integration for Hospital Information Systems (HIS).
