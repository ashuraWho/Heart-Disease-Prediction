# ============================================================ #
# SEED DATABASE - Inizializzazione Database con Dati di Test  #
# ============================================================ #
# Questo script inizializza il database SQLite con un record
# di esempio per testare le funzionalità di inferenza e query.
# Lo schema è allineato con quello definito in shared_utils.py
# ============================================================ #

from notebooks.shared_utils import init_db, get_db_connection, setup_environment, console
import pandas as pd

# Setup ambiente (seed, env vars, directories)
setup_environment()

# Inizializza database se non esiste
init_db()

# ============================================================ #
# INSERIMENTO RECORD DI TEST
# ============================================================ #
# Inserisce un record di esempio con dati clinici realistici
# Schema allineato con la definizione in shared_utils.py
# ============================================================ #

console.print("[bold cyan]Seeding database with sample patient record...[/bold cyan]")

with get_db_connection() as conn:
    # Schema allineato con shared_utils.py
    # Tutti i campi obbligatori secondo lo schema database
    conn.execute("""
        INSERT INTO patients (
            State, Sex, GeneralHealth, PhysicalHealthDays, 
            MentalHealthDays, LastCheckupTime, PhysicalActivities, 
            SleepHours, RemovedTeeth, HadHeartAttack, HadAngina, 
            HadStroke, HadAsthma, HadSkinCancer, HadCOPD, 
            HadDepressiveDisorder, HadKidneyDisease, HadArthritis, 
            HadDiabetes, DeafOrHardOfHearing, BlindOrVisionDifficulty, 
            DifficultyConcentrating, DifficultyWalking, 
            DifficultyDressingBathing, DifficultyErrands, 
            SmokerStatus, ECigaretteUsage, ChestScan, 
            RaceEthnicityCategory, AgeCategory, HeightInMeters, 
            WeightInKilograms, BMI, AlcoholDrinkers, HIVTesting, 
            FluVaxLast12, PneumoVaxEver, TetanusLast10Tdap, 
            HighRiskLastYear, CovidPos,
            Prediction, Probability
        ) VALUES (
            'California', 'Male', 'Good', 2.0, 
            1.0, 'Within past year', 'Yes', 
            7.5, 'None of them', 'No', 'No', 
            'No', 'No', 'No', 'No', 
            'No', 'No', 'No', 
            'No', 'No', 'No', 
            'No', 'No', 
            'No', 'No', 
            'Former smoker', 'Never used e-cigarettes in my entire life', 'No', 
            'White only, Non-Hispanic', 'Age 55-59', 1.75, 
            80.0, 26.12, 'Yes', 'No', 
            'No', 'No', 'No', 
            'No', 'No',
            0, 0.15
        )
    """)
    conn.commit()

console.print("[bold green]Database seeded successfully.[/bold green]")
