# ============================================================
# NOTEBOOK 1 – EDA & PREPROCESSING
# Heart Disease Prediction Dataset
# ============================================================
#
# OBIETTIVO:
# - Comprendere il dataset
# - Analizzare la distribuzione delle feature
# - Studiare la relazione con il target (HeartDisease)
# - Preparare i dati per i modelli ML/DL
#
# ============================================================

# =====================
# 1. IMPORT LIBRARIES
# =====================
# Librerie per manipolazione dati
import pandas as pd
import numpy as np

# Librerie per visualizzazione
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni estetiche dei grafici
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Librerie per preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =====================
# 2. LOAD DATASET
# =====================
# Carichiamo il dataset CSV scaricato da Kaggle
# NOTA: sostituire il path con quello corretto sul proprio sistema

data_path = "Heart_Disease_Prediction.csv"
df = pd.read_csv(data_path)

# Visualizziamo le prime righe per capire la struttura
df.head()

# =====================
# 3. DATASET OVERVIEW
# =====================
# Dimensioni del dataset
print(f"Numero di righe: {df.shape[0]}")
print(f"Numero di colonne: {df.shape[1]}")

# Tipologia delle colonne
df.info()

# =====================
# 4. RENAME COLUMNS (CLEANING)
# =====================
# Rinominiamo le colonne

rename_dict = {
    'Chest pain type': 'ChestPain', # Tipo di dolore toracico avvertito dal paziente (1-4)
    'FBS over 120': 'FBS', # Glicemia a digiuno > 120 mg/dl (1 = sì, 0 = no)
    'EKG results': 'EKG', # Risultati dell'elettrocardiogramma a riposo (0-2)
    'Max HR': 'MaxHR', # Frequenza cardiaca massima raggiunta durante l'esercizio
    'Exercise angina': 'ExerciseAngina', # Angina indotta dall'esercizio fisico (1 = sì, 0 = no)
    'ST depression': 'ST_Depression', # Depressione del tratto ST indotta dall'esercizio rispetto al riposo
    'Slope of ST': 'ST_Slope', # Pendenza del segmento ST di picco dell'esercizio (1-3)
    'Number of vessels fluro': 'NumVessels', # Numero di vasi principali (0-3) colorati tramite fluoroscopia
    'Heart Disease': 'HeartDisease' # Variabile target (1 = presenza di cardiopatia, 0 = assenza)
}

df.rename(columns=rename_dict, inplace=True)

# Verifica
print(df.columns)

# =====================
# 5. CHECK MISSING VALUES
# =====================
# In ambito medico i missing values sono critici

missing_values = df.isnull().sum()
print("\nMissing values per colonna:\n")
print(missing_values)

# In questo dataset non ci sono missing

# =====================
# 6. DUPLICATES CHECK
# =====================
# I duplicati possono introdurre bias nei modelli

duplicates = df.duplicated().sum()
print(f"\nNumero di righe duplicate: {duplicates}")

# =====================
# 7. TARGET ANALYSIS
# =====================
# Analizziamo la distribuzione del target

sns.countplot(x='HeartDisease', data=df)
plt.title('Distribuzione del Target (HeartDisease)')
plt.show()

# Percentuali
print(df['HeartDisease'].value_counts(normalize=True) * 100)

# Il dataset è leggermente sbilanciato (circa 56% vs 44%)

# =====================
# 8. NUMERICAL FEATURES ANALYSIS
# =====================
# Selezioniamo le feature numeriche continue

numerical_features = [
    'Age', 'BP', 'Cholesterol', 'MaxHR', 'ST_Depression'
]

# Distribuzione delle feature numeriche
for col in numerical_features:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribuzione di {col}')
    plt.show()

# =====================
# 9. NUMERICAL FEATURES vs TARGET
# =====================
# Boxplot per confrontare presenza/assenza di malattia

for col in numerical_features:
    sns.boxplot(x='HeartDisease', y=col, data=df)
    plt.title(f'{col} vs HeartDisease')
    plt.show()

# =====================
# 10. CATEGORICAL FEATURES ANALYSIS
# =====================
# Feature categoriali

categorical_features = [
    'Sex', 'ChestPain', 'FBS', 'EKG',
    'ExerciseAngina', 'ST_Slope', 'NumVessels', 'Thallium'
]

for col in categorical_features:
    sns.countplot(x=col, hue='HeartDisease', data=df)
    plt.title(f'{col} vs HeartDisease')
    plt.legend(title='HeartDisease')
    plt.show()

# =====================
# 11. CORRELATION ANALYSIS
# =====================
# Analisi della correlazione tra feature numeriche

# Mapping esplicito: medicalmente interpretabile
df['HeartDisease'] = df['HeartDisease'].map({
    'Absence': 0,
    'Presence': 1
})

corr_matrix = df[numerical_features + ['HeartDisease']].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# =====================
# 12. FEATURE / TARGET SPLIT
# =====================

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# =====================
# 13. TRAIN / TEST SPLIT
# =====================
# Stratificazione fondamentale per mantenere la proporzione delle classi

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =====================
# 14. PREPROCESSING PIPELINE
# =====================
# Separiamo feature numeriche e categoriali

num_features = numerical_features
cat_features = categorical_features

# Pipeline numerica:
# - Standardizzazione (media 0, varianza 1)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Pipeline categoriale:
# - OneHotEncoding (evita ordini artificiali)

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# ColumnTransformer combina le due pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# =====================
# 15. FIT & TRANSFORM
# =====================

# - fit solo sul training set
# - transform su training e test

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Shape after preprocessing (train):", X_train_processed.shape)
print("Shape after preprocessing (test):", X_test_processed.shape)
