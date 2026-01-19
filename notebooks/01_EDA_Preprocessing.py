# ============================================================ #
# Module 01 – EDA & Preprocessing (Heart 2022)                 #
# ============================================================ #

# SCOPO DEL MODULO:
# - Caricare il dataset clinico
# - Creare il target binario (HeartDisease)
# - Eseguire feature engineering semplice
# - Costruire una pipeline di preprocessing robusta
# - Salvare artefatti riutilizzabili (NO leakage)

# NOTA:
# Questo modulo NON addestra modelli.
# Serve esclusivamente a produrre dati pronti per ML/DL.

# ============================================================ #

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Aggiungiamo dinamicamente la root del progetto al PYTHONPATH
# Questo permette import puliti anche in esecuzione standalone
# ------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent))

try:
    # Import di utility condivise da tutto il progetto
    from shared_utils import (
        setup_environment, # Setup seed, env vars, stabilità runtime
        console, # Logging colorato (rich)
        DATASET_PATH, # Path assoluto al dataset
        ARTIFACTS_DIR # Directory dove salvare output intermedi
    )
except ImportError:
    # Fail fast: se shared_utils manca, il progetto NON è in stato consistente
    print("Error: shared_utils not found.")
    sys.exit(1)

# ------------------------------------------------------------ #
# Setup globale dell'ambiente:
# - seed randomici
# - env vars anti-segfault (Mac / Anaconda)
# - configurazione logging
# ------------------------------------------------------------ #
setup_environment()

# ------------------------------------------------------------ #
# Import librerie scientifiche e ML
# ------------------------------------------------------------ #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

# ------------------------------------------------------------------
# CONFIGURAZIONE GLOBALE
# ------------------------------------------------------------------
sns.set(style="whitegrid") # Stile grafici coerente
plt.rcParams["figure.figsize"] = (10, 6) # Dimensione standard figure

RANDOM_STATE = 42 # Riproducibilità
TEST_SIZE = 0.2 # Split standard 80/20

# ============================================================ #
# PARSING ARGOMENTI DA COMMAND LINE
# ============================================================ #
# Supporto per:
# --no-plots: Esegue solo preprocessing senza grafici
# --plot <tipo>: Esegue preprocessing + mostra grafico specifico
# ============================================================ #

PLOT_TYPE = None  # Tipo di grafico da mostrare
SHOW_PLOTS = True  # Se mostrare grafici (default: True)

if "--no-plots" in sys.argv:
    SHOW_PLOTS = False
elif "--plot" in sys.argv:
    try:
        plot_idx = sys.argv.index("--plot")
        if plot_idx + 1 < len(sys.argv):
            PLOT_TYPE = sys.argv[plot_idx + 1]
        SHOW_PLOTS = True
    except (ValueError, IndexError):
        console.print("[yellow]Warning: --plot specified but no plot type given[/yellow]")
        SHOW_PLOTS = False

# ------------------------------------------------------------------
# CARICAMENTO DATASET
# ------------------------------------------------------------------
console.print(f"[bold cyan]Loading dataset from:[/bold cyan] {DATASET_PATH}")

try:
    df = pd.read_csv(DATASET_PATH) # Lettura CSV clinico
except FileNotFoundError:
    # Fail esplicito se il dataset non esiste
    console.print(f"[bold red]ERROR: Dataset not found at {DATASET_PATH}[/bold red]")
    sys.exit(1)

console.print(f"Initial Rows: [bold]{len(df)}[/bold]")


# ============================================================ #
# 1. CREAZIONE DEL TARGET CLINICO
# ============================================================ #
# Definizione:
# HeartDisease = 1 se il paziente ha avuto
# - infarto (HadHeartAttack == Yes)
# - angina (HadAngina == Yes)
# ============================================================ #

console.print("[cyan]Creating Target Variable (HeartDisease)...[/cyan]")
df['HeartDisease'] = 0 # Inizializziamo tutto a "sano"

# Maschera booleana per presenza di patologia cardiaca
mask_disease = (
    (df['HadHeartAttack'] == 'Yes') |
    (df['HadAngina'] == 'Yes')
)
    
# Assegniamo 1 ai pazienti con eventi cardiaci
df.loc[mask_disease, 'HeartDisease'] = 1

# Distribuzione normalizzata del target: per visualizzare in che proporzione sono divisi i dati tra chi ha una patologia cardiaca e chi no
console.print(f"Target Distribution:\n{df['HeartDisease'].value_counts(normalize=True)}")


# ============================================================ #
# 2. FEATURE ENGINEERING LEGGERO (MA SENSATO)
# ============================================================ #

# ------------------------------------------------------------ #
# Mappiamo GeneralHealth (categorica ordinale) in numerico
# Poor -> 1 ... Excellent -> 5
# Serve per catturare una nozione di "salute percepita"
# ------------------------------------------------------------ #
gen_health_map = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very good": 4,
    "Excellent": 5
}

df['GeneralHealth_Num'] = (
    df['GeneralHealth']
    .map(gen_health_map) # Mapping esplicito
    .fillna(3) # NaN -> valore neutro "Good" (3)
)

# ------------------------------------------------------------ #
# Feature di interazione:
# Rapporto tra ore di sonno e giorni di cattiva salute fisica
# Aggiungiamo +1 per evitare divisioni per zero
# ------------------------------------------------------------ #
if 'SleepHours' in df.columns and 'PhysicalHealthDays' in df.columns:
    df['Sleep_Health_Ratio'] = df['SleepHours'] / (df['PhysicalHealthDays'] + 1)

# NOTA:
# BMI viene lasciato continuo (nessun binning)
# perché gli alberi e i modelli lineari lo gestiscono bene


# ============================================================ #
# 3. DEFINIZIONE FEATURE MATRIX X E TARGET y
# ============================================================ #

# ------------------------------------------------------------ #
# Colonne da rimuovere:
# - target leakage
# - variabili amministrative
# - campi poco interpretabili o troppo rumorosi
# ------------------------------------------------------------ #
drop_cols = [
    'State',
    'HadHeartAttack',
    'HadAngina',
    'HeartDisease',
    'LastCheckupTime',
    'RemovedTeeth',
    'TetanusLast10Tdap',
    'FluVaxLast12',
    'PneumoVaxEver',
    'HIVTesting',
    'HighRiskLastYear',
    'CovidPos'
]

# Manteniamo solo le colonne effettivamente presenti
available_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=available_cols) # Feature matrix
y = df['HeartDisease'] # Target

# ============================================================ #
# FEATURE SELECTION: RIMOZIONE FEATURE POCO INFORMATIVE
# ============================================================ #
# Rimuoviamo feature con correlazione molto bassa con il target
# per ridurre rumore e migliorare generalizzazione
# ============================================================ #

# Feature da rimuovere per correlazione troppo bassa (< 0.02)
# Identificate tramite analisi diagnostica: hanno impatto minimo sul target
weak_features = [
    'SleepHours',  # Correlazione ~0.009 (quasi zero)
    # Nota: Sleep_Health_Ratio (che usa SleepHours) ha correlazione 0.115, 
    # quindi manteniamo la feature engineered ma rimuoviamo l'originale
]

# Rimuovi solo se esistono e non sono necessarie per feature engineered
for feat in weak_features:
    if feat in X.columns:
        X = X.drop(columns=[feat])
        console.print(f"[yellow]Removed low-correlation feature: {feat}[/yellow]")

console.print(f"Feature Columns ({X.shape[1]}): {X.columns.tolist()}")


# ============================================================ #
# 4. PIPELINE DI PREPROCESSING
# ============================================================ #

# ------------------------------------------------------------ #
# Separiamo feature numeriche e categoriche
# ------------------------------------------------------------ #
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# ------------------------------------------------------------ #
# Strategia scelta:
# - StandardScaler per numeriche
# - OneHotEncoder per TUTTE le categoriche
#
# Motivazione:
# - massima robustezza in inference
# - nessun rischio di categorie nuove
# - compatibilità con modelli lineari e deep learning
# ------------------------------------------------------------ #
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ]
)


# ============================================================ #
# 5. TRAIN / TEST SPLIT + FIT PIPELINE
# ============================================================ #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

console.print("[cyan]Fitting Preprocessor...[/cyan]")
X_train_processed = preprocessor.fit_transform(X_train) # Fit SOLO sul train
X_test_processed = preprocessor.transform(X_test) # Transform sul test

console.print(f"Processed Shape: {X_train_processed.shape}")


# ============================================================ #
# 6. SALVATAGGIO ARTEFATTI
# ============================================================ #
# Tutto ciò che serve ai moduli successivi:
# - preprocessor
# - dati processati
# - target
# - feature names (per inference & explainability)
# ============================================================ #

dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

np.save(ARTIFACTS_DIR / "y_train.npy", y_train)
np.save(ARTIFACTS_DIR / "y_test.npy", y_test)

np.savez(ARTIFACTS_DIR / "X_train.npz", X=X_train_processed)
np.savez(ARTIFACTS_DIR / "X_test.npz", X=X_test_processed)

dump(X.columns.tolist(), ARTIFACTS_DIR / "feature_names.joblib")

console.print(f"[bold green]Artifacts saved to {ARTIFACTS_DIR}[/bold green]")

# ============================================================ #
# 7. VISUALIZZAZIONI (OPZIONALI)
# ============================================================ #
# Funzioni per generare grafici EDA specifici
# ============================================================ #

def plot_correlation_matrix():
    """Visualizza correlation matrix delle feature numeriche"""
    console.print("\n[bold cyan]Generating Correlation Matrix...[/bold cyan]")
    
    # Feature numeriche dal dataset originale (prima del preprocessing)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) < 2:
        console.print("[yellow]Not enough numeric features for correlation matrix[/yellow]")
        return
    
    # Calcola correlazioni
    corr_matrix = X[numeric_features].corr()
    
    # Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Feature Correlation Matrix (Numeric Features)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved to {ARTIFACTS_DIR / 'correlation_matrix.png'}[/green]")
    plt.show()


def plot_target_distribution():
    """Visualizza distribuzione del target"""
    console.print("\n[bold cyan]Generating Target Distribution Plot...[/bold cyan]")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribuzione conteggio
    counts = y.value_counts()
    ax1.bar(['Healthy (0)', 'Disease (1)'], counts.values, color=['green', 'red'], alpha=0.7)
    ax1.set_title('Target Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori sopra le barre
    for i, v in enumerate(counts.values):
        ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Distribuzione percentuale
    percentages = y.value_counts(normalize=True) * 100
    ax2.bar(['Healthy (0)', 'Disease (1)'], percentages.values, color=['green', 'red'], alpha=0.7)
    ax2.set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage (%)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori sopra le barre
    for i, v in enumerate(percentages.values):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "target_distribution.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved to {ARTIFACTS_DIR / 'target_distribution.png'}[/green]")
    plt.show()


def plot_feature_vs_target():
    """Visualizza top feature in relazione al target"""
    console.print("\n[bold cyan]Generating Feature vs Target Plots...[/bold cyan]")
    
    # Top 6 feature categoriche più correlate
    top_categorical = ['GeneralHealth', 'AgeCategory', 'ChestScan', 'DifficultyWalking', 'HadStroke', 'HadDiabetes']
    available_cat = [f for f in top_categorical if f in X.columns][:6]
    
    if not available_cat:
        console.print("[yellow]No suitable categorical features found[/yellow]")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feat in enumerate(available_cat):
        ax = axes[idx]
        
        # Crosstab feature vs target
        crosstab = pd.crosstab(X[feat], y, normalize='index') * 100
        
        # Plot stacked bar
        crosstab.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title(f'{feat} vs Heart Disease', fontsize=12, fontweight='bold')
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.legend(['Healthy', 'Disease'], fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "feature_vs_target.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved to {ARTIFACTS_DIR / 'feature_vs_target.png'}[/green]")
    plt.show()


def plot_age_health_analysis():
    """Analisi età e salute generale"""
    console.print("\n[bold cyan]Generating Age & Health Analysis...[/bold cyan]")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Età vs Malattia
    if 'AgeCategory' in X.columns:
        age_disease = pd.crosstab(X['AgeCategory'], y, normalize='index') * 100
        age_disease.plot(kind='barh', stacked=True, ax=axes[0, 0], color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('Age Category vs Heart Disease', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Percentage (%)')
        axes[0, 0].legend(['Healthy', 'Disease'], fontsize=9)
        axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Salute generale vs Malattia
    if 'GeneralHealth' in X.columns:
        health_disease = pd.crosstab(X['GeneralHealth'], y, normalize='index') * 100
        health_disease.plot(kind='bar', stacked=True, ax=axes[0, 1], color=['green', 'red'], alpha=0.7)
        axes[0, 1].set_title('General Health vs Heart Disease', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('General Health')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].legend(['Healthy', 'Disease'], fontsize=9)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Physical Health Days (distribuzione)
    if 'PhysicalHealthDays' in X.columns:
        X[y == 0]['PhysicalHealthDays'].hist(ax=axes[1, 0], bins=30, alpha=0.6, label='Healthy', color='green')
        X[y == 1]['PhysicalHealthDays'].hist(ax=axes[1, 0], bins=30, alpha=0.6, label='Disease', color='red')
        axes[1, 0].set_title('Physical Health Days Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. BMI vs Malattia
    if 'BMI' in X.columns:
        X[y == 0]['BMI'].hist(ax=axes[1, 1], bins=30, alpha=0.6, label='Healthy', color='green')
        X[y == 1]['BMI'].hist(ax=axes[1, 1], bins=30, alpha=0.6, label='Disease', color='red')
        axes[1, 1].set_title('BMI Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('BMI')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "age_health_analysis.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved to {ARTIFACTS_DIR / 'age_health_analysis.png'}[/green]")
    plt.show()


def plot_risk_factors():
    """Analisi fattori di rischio"""
    console.print("\n[bold cyan]Generating Risk Factors Analysis...[/bold cyan]")
    
    risk_factors = ['HadDiabetes', 'HadStroke', 'HadCOPD', 'SmokerStatus', 'PhysicalActivities']
    available_risks = [f for f in risk_factors if f in X.columns]
    
    if not available_risks:
        console.print("[yellow]No risk factor features found[/yellow]")
        return
    
    fig, axes = plt.subplots(1, len(available_risks), figsize=(5*len(available_risks), 6))
    if len(available_risks) == 1:
        axes = [axes]
    
    for idx, risk in enumerate(available_risks):
        ax = axes[idx]
        
        # Calcola percentuale malattia per ogni livello del fattore di rischio
        risk_disease = pd.crosstab(X[risk], y, normalize='index') * 100
        
        risk_disease.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7, width=0.8)
        ax.set_title(f'{risk}\nDisease Rate', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Healthy', 'Disease'], fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "risk_factors.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved to {ARTIFACTS_DIR / 'risk_factors.png'}[/green]")
    plt.show()


def plot_top_correlations():
    """Visualizza top correlazioni feature-target"""
    console.print("\n[bold cyan]Generating Top Correlations Plot...[/bold cyan]")
    
    # Carica analisi correlazioni se disponibile
    corr_path = ARTIFACTS_DIR.parent / "artifacts" / "feature_correlations.csv"
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        top_15 = corr_df.head(15)
    else:
        console.print("[yellow]Correlation analysis file not found. Generating correlations...[/yellow]")
        # Calcola correlazioni al volo
        from scipy.stats import pointbiserialr, chi2_contingency
        correlations = []
        for col in X.columns:
            if X[col].dtype in [np.int64, np.float64]:
                try:
                    corr, _ = pointbiserialr(X[col], y)
                    correlations.append({'Feature': col, 'Correlation': abs(corr)})
                except:
                    pass
        if correlations:
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
            top_15 = corr_df.head(15)
        else:
            console.print("[red]Could not calculate correlations[/red]")
            return
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0.1 else 'orange' if x < 0.15 else 'green' for x in top_15['Correlation']]
    plt.barh(range(len(top_15)), top_15['Correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_15)), top_15['Feature'])
    plt.xlabel('Correlation with Target (Absolute)', fontsize=12)
    plt.title('Top 15 Features - Correlation with Heart Disease', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # Aggiungi valori
    for i, v in enumerate(top_15['Correlation']):
        plt.text(v, i, f' {v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "top_correlations.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved to {ARTIFACTS_DIR / 'top_correlations.png'}[/green]")
    plt.show()


# Fine del modulo


# ============================================================ #
# ESECUZIONE GRAFICI (se richiesti)
# ============================================================ #

if SHOW_PLOTS and PLOT_TYPE:
    # Esegue solo il grafico richiesto
    plot_functions = {
        'correlation': plot_correlation_matrix,
        'target': plot_target_distribution,
        'features': plot_feature_vs_target,
        'age_health': plot_age_health_analysis,
        'risk_factors': plot_risk_factors,
        'top_correlations': plot_top_correlations
    }
    
    if PLOT_TYPE in plot_functions:
        try:
            plot_functions[PLOT_TYPE]()
        except Exception as e:
            console.print(f"[bold red]Error generating plot: {e}[/bold red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    else:
        console.print(f"[yellow]Unknown plot type: {PLOT_TYPE}[/yellow]")
        console.print(f"[cyan]Available plot types: {', '.join(plot_functions.keys())}[/cyan]")
