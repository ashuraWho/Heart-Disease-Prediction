import pandas as pd
import numpy as np
from notebooks.shared_utils import DATASET_PATH, setup_environment
from rich.console import Console
from rich.table import Table

setup_environment()
console = Console()

df = pd.read_csv(DATASET_PATH)

console.print(f"[bold]Total Rows:[/bold] {len(df)}")
console.print(f"[bold]Duplicates:[/bold] {df.duplicated().sum()}")

# Missing Values
missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    console.print("\n[bold red]Missing Values:[/bold red]")
    console.print(missing)

# Target Distribution
if "Heart Disease Status" in df.columns:
    target = df["Heart Disease Status"].value_counts(normalize=True)
    console.print("\n[bold]Target Distribution:[/bold]")
    console.print(target)

# Numerical Stats
console.print("\n[bold]Numerical Stats:[/bold]")
print(df.describe())
