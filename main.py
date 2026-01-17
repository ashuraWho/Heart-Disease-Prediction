# ============================================================ #
# Heart Disease Prediction - Integrated CLI Dashboard          #
# MAIN MODULE – MEDICAL DECISION SUPPORT SYSTEM                #
# ============================================================ #

import os
import sys
import subprocess
from pathlib import Path

# Add notebooks directory to sys.path to allow importing shared_utils
sys.path.append(str(Path(__file__).resolve().parent / "notebooks"))

try:
    from shared_utils import (
        setup_environment, 
        print_header, 
        clear_screen, 
        NOTEBOOKS_DIR, 
        ARTIFACTS_DIR, 
        DB_PATH,
        console
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import shared_utils. Make sure you are running from the project root. {e}")
    sys.exit(1)

from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

# --- INITIALIZE ENVIRONMENT ---
setup_environment()

# ------------------------------

def run_module(script_name, args=None):
    """Runs a specific module script using subprocess."""
    script_path = NOTEBOOKS_DIR / script_name
    console.print(f"\n[bold cyan][EXECUTION] Starting {script_name}...[/bold cyan]")
    
    # Pass current environment variables to the subprocess
    env = os.environ.copy()
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
        
    try:
        subprocess.run(cmd, check=True, env=env)
        console.print(f"\n[bold green][SUCCESS] {script_name} completed successfully.[/bold green]")
    except subprocess.CalledProcessError as e:
        if e.returncode == -11 or e.returncode == 139: # SegFaults
             console.print(Panel(f"[bold red][CRITICAL ERROR] Module {script_name} experienced a Segmentation Fault.[/bold red]\n\n[yellow]>>> Please ensure you are NOT in the 'base' environment and use 'setup_mac.sh'.[/yellow]", title="Execution Error"))
        else:
             console.print(f"\n[bold red][ERROR] Module {script_name} failed with exit code {e.returncode}.[/bold red]")

def show_clinical_guide():
    """Displays clinical parameter explanations using Rich."""
    clear_screen()
    print_header("CLINICAL DATA GLOSSARY & GUIDE")
    
    from shared_utils import CLINICAL_GUIDE
    
    for key, value in CLINICAL_GUIDE.items():
        console.print(f"[bold cyan]► {key.ljust(20)}[/bold cyan]: {value}")
        
    Prompt.ask("\nPress Enter to return to main menu", show_default=False)

def eda_interactive_menu():
    """Sub-menu for EDA plots."""
    while True:
        clear_screen()
        print_header("EDA & DATA VISUALIZATION")
        console.print("[1] Show Correlation Matrix (Numerical Heatmap)")
        console.print("[2] Show Target Variable Distribution")
        console.print("[3] Show Individual Feature Plots (One by One)")
        console.print("[q] Return to Main Menu")
        
        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "q"])
        
        if choice == '1': run_module("01_EDA_Preprocessing.py", args=["--plots"])
        elif choice == '2': run_module("01_EDA_Preprocessing.py", args=["--plots"])
        elif choice == '3': run_module("01_EDA_Preprocessing.py", args=["--plots"])
        elif choice == 'q': break

def reset_artifacts():
    """Clears all generated artifacts."""
    if ARTIFACTS_DIR.exists():
        import shutil
        shutil.rmtree(ARTIFACTS_DIR)
        ARTIFACTS_DIR.mkdir()
        console.print(Panel("\n[bold green][RESET] All artifacts deleted.[/bold green]\n[yellow]You MUST run Module 01 & 02 again.[/yellow]", title="System Maintenance"))
    else:
        console.print("\n[INFO] No artifacts folder found.")

def delete_database():
    """Deletes the SQL database."""
    if DB_PATH.exists():
        os.remove(DB_PATH)
        console.print(f"\n[bold red][DELETE] Database '{DB_PATH.name}' deleted.[/bold red]")
    else:
        console.print("\n[INFO] No database found.")

def main_menu():
    """Main menu loop."""
    while True:
        clear_screen()
        print_header("HEART DISEASE PREDICTION SYSTEM")
        
        console.print(Panel.fit(
            "[bold white]1.[/bold white] [cyan][Data][/cyan] EDA & Visual Analysis (Interactive Plots)\n"
            "[bold white]2.[/bold white] [cyan][Training][/cyan] Unified Model Competition (ML vs DL)\n"
            "[bold white]3.[/bold white] [cyan][XAI][/cyan] Explainability (SHAP Analysis)\n"
            "[bold white]4.[/bold white] [cyan][Patient][/cyan] Predict for a New Patient (Manual Entry)\n"
            "[bold white]5.[/bold white] [cyan][History][/cyan] Batch Predict all Patients in Database\n"
            "[bold white]6.[/bold white] [cyan][Knowledge][/cyan] Clinical Glossary (Definitions)\n\n"
            "[bold white]d.[/bold white] [red][Maintenance][/red] Delete SQL Patient Database\n"
            "[bold white]r.[/bold white] [red][Maintenance][/red] Reset ML Artifacts\n"
            "[bold white]q.[/bold white] [white]Exit System[/white]",
            title="Main Menu",
            border_style="blue"
        ))
        
        console.print("[italic dim]Tip: Run Module 1 & 2 first to enable Predictions and XAI.[/italic dim]")
        
        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4", "5", "6", "d", "r", "q"])
        
        if choice == '1': eda_interactive_menu()
        elif choice == '2': 
            run_module("02_Unified_Training.py")
            Prompt.ask("\nPress Enter to continue", show_default=False)
        elif choice == '3':
            run_module("03_Explainability.py")
            Prompt.ask("\nPress Enter to continue", show_default=False)
        elif choice == '4':
            run_module("04_Inference.py")
            Prompt.ask("\nPress Enter to continue", show_default=False)
        elif choice == '5':
            run_module("04_Inference.py", args=["--batch"])
            Prompt.ask("\nPress Enter to continue", show_default=False)
        elif choice == '6': show_clinical_guide()
        elif choice == 'd': 
            delete_database()
            Prompt.ask("\nPress Enter to continue", show_default=False)
        elif choice == 'r': 
            reset_artifacts()
            Prompt.ask("\nPress Enter to continue", show_default=False)
        elif choice == 'q':
            console.print("[bold green]Exiting. Stay healthy![/bold green]")
            break

if __name__ == "__main__":
    main_menu()
