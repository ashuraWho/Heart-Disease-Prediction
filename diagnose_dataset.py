# ============================================================ #
# DATASET DIAGNOSTIC TOOL
# ============================================================ #
# Analizza la qualità del dataset per identificare problemi:
# - Feature poco informative
# - Correlazioni con il target
# - Potenziale leakage
# - Distribuzioni problematiche
# ============================================================ #

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "notebooks"))

from shared_utils import setup_environment, console, DATASET_PATH
setup_environment()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, pointbiserialr
import warnings
warnings.filterwarnings('ignore')

console.print("[bold cyan]=== DATASET DIAGNOSTIC ANALYSIS ===[/bold cyan]\n")

# Carica dataset
df = pd.read_csv(DATASET_PATH)

# Crea target
df['HeartDisease'] = ((df['HadHeartAttack'] == 'Yes') | (df['HadAngina'] == 'Yes')).astype(int)

console.print(f"[bold]Dataset Shape:[/bold] {df.shape}")
console.print(f"[bold]Target Distribution:[/bold]")
console.print(f"  Healthy (0): {sum(df['HeartDisease'] == 0)} ({sum(df['HeartDisease'] == 0)/len(df)*100:.1f}%)")
console.print(f"  Disease (1): {sum(df['HeartDisease'] == 1)} ({sum(df['HeartDisease'] == 1)/len(df)*100:.1f}%)")
console.print(f"  Imbalance Ratio: {sum(df['HeartDisease'] == 0) / sum(df['HeartDisease'] == 1):.2f}:1\n")

# Feature engineering (come nel modulo 01)
gen_health_map = {"Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5}
if 'GeneralHealth' in df.columns:
    df['GeneralHealth_Num'] = df['GeneralHealth'].map(gen_health_map).fillna(3)
if 'SleepHours' in df.columns and 'PhysicalHealthDays' in df.columns:
    df['Sleep_Health_Ratio'] = df['SleepHours'] / (df['PhysicalHealthDays'] + 1)

# Rimuovi colonne target e amministrative
drop_cols = [
    'State', 'HadHeartAttack', 'HadAngina', 'HeartDisease',
    'LastCheckupTime', 'RemovedTeeth', 'TetanusLast10Tdap',
    'FluVaxLast12', 'PneumoVaxEver', 'HIVTesting', 'HighRiskLastYear', 'CovidPos'
]
available_drop = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=available_drop)
y = df['HeartDisease']

console.print(f"[bold]Features available:[/bold] {X.shape[1]}\n")

# ============================================================ #
# ANALISI CORRELAZIONE FEATURE-TARGET
# ============================================================ #

console.print("[bold yellow]=== FEATURE-TARGET CORRELATIONS ===[/bold yellow]\n")

correlations = []

for col in X.columns:
    if X[col].dtype in [np.int64, np.float64]:
        # Correlazione numerica (point-biserial)
        try:
            corr, p_val = pointbiserialr(X[col], y)
            correlations.append({
                'Feature': col,
                'Type': 'Numeric',
                'Correlation': abs(corr),
                'P-Value': p_val,
                'Significant': p_val < 0.05
            })
        except:
            pass
    else:
        # Correlazione categorica (Cramér's V)
        try:
            contingency = pd.crosstab(X[col], y)
            chi2, p_val, _, _ = chi2_contingency(contingency)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            correlations.append({
                'Feature': col,
                'Type': 'Categorical',
                'Correlation': cramers_v,
                'P-Value': p_val,
                'Significant': p_val < 0.05
            })
        except:
            pass

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('Correlation', ascending=False)

# Top 20 feature più correlate
console.print(f"[bold]Top 20 Most Correlated Features:[/bold]")
top_20 = corr_df.head(20)
for idx, row in top_20.iterrows():
    sig = "[green]✓[/green]" if row['Significant'] else "[red]✗[/red]"
    console.print(f"  {sig} {row['Feature']:<30} {row['Type']:<12} r={row['Correlation']:.4f} (p={row['P-Value']:.4f})")

console.print(f"\n[bold]Features with correlation > 0.1:[/bold] {len(corr_df[corr_df['Correlation'] > 0.1])}")
console.print(f"[bold]Features with correlation > 0.05:[/bold] {len(corr_df[corr_df['Correlation'] > 0.05])}")
console.print(f"[bold]Features with correlation < 0.01:[/bold] {len(corr_df[corr_df['Correlation'] < 0.01])}")

# Features poco informative
weak_features = corr_df[corr_df['Correlation'] < 0.01]
if len(weak_features) > 0:
    console.print(f"\n[yellow]⚠️ WARNING: {len(weak_features)} features have very weak correlation (< 0.01) with target[/yellow]")
    console.print("[dim]Consider removing these features:[/dim]")
    for idx, row in weak_features.iterrows():
        console.print(f"  - {row['Feature']} (r={row['Correlation']:.6f})")

# ============================================================ #
# ANALISI VARIABILITÀ
# ============================================================ #

console.print(f"\n[bold yellow]=== FEATURE VARIANCE ANALYSIS ===[/bold yellow]\n")

# Features numeriche con varianza quasi zero (costanti o quasi)
numeric_cols = X.select_dtypes(include=[np.number]).columns
low_var_features = []
for col in numeric_cols:
    std = X[col].std()
    if std < 0.01:
        low_var_features.append(col)

if low_var_features:
    console.print(f"[yellow]⚠️ Features with very low variance (near constant):[/yellow]")
    for col in low_var_features:
        console.print(f"  - {col} (std={X[col].std():.6f})")
else:
    console.print("[green]✓ No constant/near-constant numeric features found[/green]")

# ============================================================ #
# ANALISI MISSING VALUES
# ============================================================ #

console.print(f"\n[bold yellow]=== MISSING VALUES ANALYSIS ===[/bold yellow]\n")

missing = X.isnull().sum()
missing_pct = (missing / len(X)) * 100
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing_Count': missing.values,
    'Missing_Pct': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Pct', ascending=False)

if len(missing_df) > 0:
    console.print(f"[yellow]⚠️ Features with missing values:[/yellow]")
    for _, row in missing_df.iterrows():
        console.print(f"  - {row['Feature']}: {row['Missing_Count']} ({row['Missing_Pct']:.2f}%)")
else:
    console.print("[green]✓ No missing values (as expected from 'no_nans' dataset)[/green]")

# ============================================================ #
# RACCOMANDAZIONI
# ============================================================ #

console.print(f"\n[bold cyan]=== RECOMMENDATIONS ===[/bold cyan]\n")

strong_features = corr_df[corr_df['Correlation'] > 0.1]
console.print(f"[bold]Strong predictive features (r > 0.1):[/bold] {len(strong_features)}")
if len(strong_features) < 10:
    console.print(f"[red]⚠️ CRITICAL: Only {len(strong_features)} features have strong correlation with target![/red]")
    console.print("[yellow]This might explain poor model performance.[/yellow]")
    console.print("[cyan]Consider:[/cyan]")
    console.print("  1. Advanced feature engineering")
    console.print("  2. Domain expert consultation for feature selection")
    console.print("  3. External data sources to enrich features")

console.print(f"\n[bold]Summary:[/bold]")
console.print(f"  Total features: {len(corr_df)}")
console.print(f"  Strong (r>0.1): {len(corr_df[corr_df['Correlation'] > 0.1])}")
console.print(f"  Moderate (0.05<r<0.1): {len(corr_df[(corr_df['Correlation'] > 0.05) & (corr_df['Correlation'] <= 0.1)])}")
console.print(f"  Weak (r<0.05): {len(corr_df[corr_df['Correlation'] < 0.05])}")

# Salva risultati
output_path = Path(__file__).parent / "artifacts" / "feature_correlations.csv"
output_path.parent.mkdir(exist_ok=True)
corr_df.to_csv(output_path, index=False)
console.print(f"\n[green]✓ Correlation analysis saved to: {output_path}[/green]")
