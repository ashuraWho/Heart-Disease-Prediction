# ============================================================ #
# Heart Disease Prediction - Integrated CLI Dashboard          #
# MAIN MODULE – MEDICAL DECISION SUPPORT SYSTEM                #
# ============================================================ #
# Questo è il punto di ingresso dell'intero progetto.
# Funziona come dashboard CLI interattiva che orchestra:
# - EDA
# - Training
# - Explainability (XAI)
# - Inference clinica
# - Gestione database e manutenzione
# ============================================================ #

import os
import sys
import subprocess
from pathlib import Path

# ------------------------------------------------------------
# Aggiunge la directory "notebooks" al PYTHONPATH
# Serve per permettere a questo modulo di importare shared_utils
# anche se viene eseguito dalla root del progetto
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent / "notebooks"))

# Import delle utility condivise
try:
    from shared_utils import (
        setup_environment, # Setup globale (console, seed, dirs)
        print_header, # Stampa header ASCII stilizzato
        clear_screen, # Clear screen cross-platform
        NOTEBOOKS_DIR, # Path alla cartella notebooks/
        ARTIFACTS_DIR, # Path alla cartella artifacts/
        DB_PATH, # Path al database SQLite
        console, # Console Rich
        check_models_exist # Verifica se i modelli sono già addestrati
    )
except ImportError as e:
    # Errore critico: il sistema non può funzionare senza shared_utils
    print(f"CRITICAL ERROR: Could not import shared_utils. Make sure you are running from the project root. {e}")
    sys.exit(1)

# Componenti Rich per UI CLI
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

# ------------------------------------------------------------
# INIZIALIZZAZIONE AMBIENTE
# ------------------------------------------------------------
# Imposta:
# - stile Rich
# - directory
# - eventuali seed globali
setup_environment()


# ============================================================ #
# UTILITY: ESECUZIONE MODULI
# ============================================================ #
# Esegue un modulo (01,02,03,04) come processo separato
# -> isolamento memoria
# -> evita conflitti TensorFlow / SHAP / matplotlib
# ============================================================ #

def run_module(script_name, args=None):
    
    # Costruisce il path assoluto allo script
    script_path = NOTEBOOKS_DIR / script_name
    console.print(f"\n[bold cyan][EXECUTION] Starting {script_name}...[/bold cyan]")
    
    # Copia le variabili d'ambiente correnti
    env = os.environ.copy()
    
    # Comando: python script.py [args]
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
        
    try:
        # Esecuzione bloccante del modulo
        subprocess.run(cmd, check=True, env=env)
        console.print(f"\n[bold green][SUCCESS] {script_name} completed successfully.[/bold green]")
        
    except subprocess.CalledProcessError as e:
        # Gestione specifica segmentation fault (molto comune con TF/SHAP)
        if e.returncode == -11 or e.returncode == 139: # SegFaults
             console.print(Panel(f"[bold red][CRITICAL ERROR] Module {script_name} experienced a Segmentation Fault.[/bold red]\n\n[yellow]>>> Please ensure you are NOT in the 'base' environment and use 'setup_mac.sh'.[/yellow]", title="Execution Error"))
        else:
             console.print(f"\n[bold red][ERROR] Module {script_name} failed with exit code {e.returncode}.[/bold red]")


# ============================================================ #
# CLINICAL GUIDE
# ============================================================ #
# Visualizza spiegazioni cliniche delle feature
# ============================================================ #

def show_clinical_guide():
    
    clear_screen()
    print_header("CLINICAL DATA GLOSSARY & GUIDE")
    
    # Import locale per evitare cicli
    from shared_utils import CLINICAL_GUIDE
    
    # Stampa ogni parametro clinico
    for key, value in CLINICAL_GUIDE.items():
        console.print(f"[bold cyan]► {key.ljust(20)}[/bold cyan]: {value}")
        
    Prompt.ask("\nPress Enter to return to main menu", show_default=False)


# ============================================================ #
# EDA INTERACTIVE MENU
# ============================================================ #
# Sottomenu dedicato alla visualizzazione dati
# Prima esegue preprocessing, poi mostra menu grafici
# ============================================================ #

def eda_interactive_menu():
    """
    Menu EDA che:
    1. Esegue automaticamente preprocessing/feature engineering
    2. Mostra sottomenu per visualizzazioni
    """
    
    # STEP 1: Esegui preprocessing automaticamente
    console.print("[bold cyan]Running preprocessing and feature engineering...[/bold cyan]")
    run_module("01_EDA_Preprocessing.py", args=["--no-plots"])
    
    # STEP 2: Menu visualizzazioni
    while True:
        clear_screen()
        print_header("EDA & DATA VISUALIZATION")
        console.print(Panel.fit(
            "[bold white]1.[/bold white] [cyan]Correlation Matrix[/cyan] - Heatmap correlazioni numeriche\n"
            "[bold white]2.[/bold white] [cyan]Target Distribution[/cyan] - Distribuzione classe target\n"
            "[bold white]3.[/bold white] [cyan]Feature vs Target[/cyan] - Grafici feature per classe\n"
            "[bold white]4.[/bold white] [cyan]Age & Health Analysis[/cyan] - Analisi età e salute\n"
            "[bold white]5.[/bold white] [cyan]Disease Risk Factors[/cyan] - Fattori di rischio malattia\n"
            "[bold white]6.[/bold white] [cyan]Feature Correlations[/cyan] - Top correlazioni con target\n\n"
            "[bold white]q.[/bold white] [white]Return to Main Menu[/white]",
            title="Visualization Options",
            border_style="blue"
        ))
        
        choice = Prompt.ask("\nSelect a visualization", choices=["1", "2", "3", "4", "5", "6", "q"])
        
        # Chiama modulo 01 con flag specifico per ogni tipo di grafico
        if choice == '1': 
            run_module("01_EDA_Preprocessing.py", args=["--plot", "correlation"])
        elif choice == '2': 
            run_module("01_EDA_Preprocessing.py", args=["--plot", "target"])
        elif choice == '3': 
            run_module("01_EDA_Preprocessing.py", args=["--plot", "features"])
        elif choice == '4': 
            run_module("01_EDA_Preprocessing.py", args=["--plot", "age_health"])
        elif choice == '5': 
            run_module("01_EDA_Preprocessing.py", args=["--plot", "risk_factors"])
        elif choice == '6': 
            run_module("01_EDA_Preprocessing.py", args=["--plot", "top_correlations"])
        elif choice == 'q': 
            break


# ============================================================ #
# MAINTENANCE: RESET ARTIFACTS
# ============================================================ #

def reset_artifacts():

    if ARTIFACTS_DIR.exists():
        import shutil
        
        # Cancella e ricrea la cartella artifacts
        shutil.rmtree(ARTIFACTS_DIR)
        ARTIFACTS_DIR.mkdir()
        
        console.print(Panel("\n[bold green][RESET] All artifacts deleted.[/bold green]\n[yellow]You MUST run Module 01 & 02 again.[/yellow]", title="System Maintenance"))
    else:
        console.print("\n[INFO] No artifacts folder found.")


# ============================================================ #
# MAINTENANCE: DELETE DATABASE
# ============================================================ #

def delete_database():

    if DB_PATH.exists():
        os.remove(DB_PATH)
        console.print(f"\n[bold red][DELETE] Database '{DB_PATH.name}' deleted.[/bold red]")
    else:
        console.print("\n[INFO] No database found.")


# ============================================================ #
# MAIN MENU LOOP
# ============================================================ #
# Dashboard principale del sistema
# ============================================================ #

def main_menu():

    while True:
        clear_screen()
        print_header("HEART DISEASE PREDICTION SYSTEM")
        
        # Pannello principale con tutte le funzionalità
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
        
        # Routing logico delle opzioni
        if choice == '1': eda_interactive_menu()
        elif choice == '2': 
            # ============================================================
            # GESTIONE MODELLI ESISTENTI - Sistema di salvataggio/caricamento
            # ============================================================
            # Prima di avviare il training, verifica se esistono già modelli
            # addestrati da una sessione precedente. Se esistono, offre all'utente
            # la possibilità di:
            # 1. Usare i modelli esistenti (skip training) - utile per risparmiare tempo
            # 2. Rifare il training da zero (sovrascrive i modelli esistenti)
            #
            # Questo permette all'utente di:
            # - Continuare a lavorare con modelli già addestrati dopo aver chiuso il programma
            # - Evitare di rifare il training ogni volta (training può richiedere 10-30 minuti)
            # - Mantenere traccia dei modelli tra diverse sessioni di lavoro
            # ============================================================
            
            # Verifica presenza modelli addestrati completi
            if check_models_exist():
                # Modelli trovati: mostra menu di scelta
                console.print(Panel(
                    "[bold cyan]✓ Modelli addestrati trovati![/bold cyan]\n\n"
                    "È stato trovato un training precedente completo.\n"
                    "Puoi scegliere di:\n"
                    "  • Usare i modelli esistenti (skip training)\n"
                    "  • Rifare il training (sovrascriverà i modelli esistenti)",
                    title="Training Options",
                    border_style="cyan"
                ))
                
                # Richiedi scelta all'utente
                train_choice = Prompt.ask(
                    "\nVuoi usare i modelli esistenti o rifare il training?",
                    choices=["use", "retrain"],
                    default="use"
                )
                
                if train_choice == "use":
                    # Scelta: usare modelli esistenti
                    # Non eseguire training, i modelli sono già disponibili in artifacts/
                    # I moduli 03 (Explainability) e 04 (Inference) caricheranno automaticamente
                    # i modelli da artifacts/ quando necessario
                    console.print("\n[bold green]✓ Usando modelli esistenti. Training saltato.[/bold green]")
                    console.print("[cyan]I modelli sono pronti per inference e explainability.[/cyan]")
                else:  # retrain
                    # Scelta: rifare training da zero
                    # Avvia il training che sovrascriverà i modelli esistenti in artifacts/
                    console.print("\n[bold yellow]⚠️ Avvio nuovo training. I modelli esistenti saranno sovrascritti.[/bold yellow]")
                    run_module("02_Unified_Training.py")
            else:
                # Nessun modello esistente trovato
                # Procedi normalmente con il training (primo training o dopo reset artifacts)
                console.print("[cyan]Nessun modello trovato. Avvio training...[/cyan]")
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


# ============================================================ #
# ENTRY POINT
# ============================================================ #

if __name__ == "__main__":
    main_menu()
