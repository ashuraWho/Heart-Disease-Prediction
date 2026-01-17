from notebooks.shared_utils import init_db, get_db_connection, setup_environment
import pandas as pd

setup_environment()
init_db()

# Insert one record
with get_db_connection() as conn:
    # Schema: Age, Gender, [Blood Pressure], [Cholesterol Level], ...
    conn.execute("""
        INSERT INTO patients (
            Age, Gender, "Blood Pressure", "Cholesterol Level", 
            "Exercise Habits", Smoking, "Family Heart Disease", Diabetes, BMI,
            "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
            "Alcohol Consumption", "Stress Level", "Sleep Hours", "Sugar Consumption",
            "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
            Prediction, Probability
        ) VALUES (
            55, 'Male', 120, 200, 
            'Medium', 'No', 'No', 'No', 25.0,
            'No', 'No', 'No', 
            'Low', 'Low', 7, 'Low', 
            150, 90, 1.0, 10.0,
            0, 0.1
        )
    """)
    conn.commit()

print("DB Seeded.")
