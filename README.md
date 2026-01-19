# 🫀 Sistema di Predizione Malattie Cardiache con Machine Learning

## 📋 Indice

1. [Panoramica del Progetto](#panoramica-del-progetto)
2. [Struttura del Progetto](#struttura-del-progetto)
3. [Architettura e Decisioni Progettuali](#architettura-e-decisioni-progettuali)
4. [Moduli del Sistema](#moduli-del-sistema)
5. [Dataset e Limitazioni](#dataset-e-limitazioni)
6. [Installazione e Configurazione](#installazione-e-configurazione)
7. [Utilizzo del Sistema](#utilizzo-del-sistema)
8. [Manutenzione](#manutenzione)
9. [Risultati Attesi e Performance](#risultati-attesi-e-performance)
10. [Roadmap e Miglioramenti Futuri](#roadmap-e-miglioramenti-futuri)

---

## 🎯 Panoramica del Progetto

### Missione

Questo progetto implementa un **Clinical Decision Support System (CDSS)** completo per la predizione di malattie cardiache basato su dati clinici. Il sistema combina **Machine Learning classico** e **Deep Learning** in un framework competitivo unificato, fornendo:

- **Predizione**: Classificazione binaria (presenza/assenza di malattia cardiaca)
- **Explainability**: Analisi SHAP per interpretare le decisioni del modello
- **Persistenza**: Database SQLite per storico pazienti
- **Interfaccia CLI**: Dashboard interattiva per clinici e ricercatori (Command Line Interface)

### Caratteristiche Principali

- **Ensemble di 8 modelli**: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, Neural Network
- **Bilanciamento classi**: Strategia combinata Undersampling + SMOTE per dataset estremamente sbilanciati
- **Ottimizzazione threshold**: Tournament di metriche per selezionare la soglia decisionale ottimale
- **Explainability**: Analisi SHAP globale e locale per ogni predizione
- **Database persistente**: SQLite per tracciamento storico pazienti
- **Salvataggio/Caricamento modelli**: Sistema automatico per salvare e riutilizzare modelli addestrati tra diverse sessioni

---

## 📂 Struttura del Progetto

```
heart-disease-prediction/
│
├── main.py                          # Entry point - Dashboard CLI principale
├── README.md                        # Documentazione completa (questo file)
├── requirements.txt                 # Dipendenze Python
│
├── notebooks/                       # Moduli principali del sistema
│   ├── shared_utils.py             # Utility condivise (config, DB, console)
│   ├── 01_EDA_Preprocessing.py     # EDA e preprocessing dati
│   ├── 02_Unified_Training.py      # Training ensemble modelli
│   ├── 03_Explainability.py        # Analisi SHAP e interpretabilità
│   └── 04_Inference.py             # Predizione pazienti (interattiva e batch)
│
├── data/                            # Dataset
│   └── heart_2022_no_nans.csv      # Dataset CDC 2022 (246 022 record) - [Link Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
│
├── artifacts/                       # Output del sistema (generati automaticamente)
│   ├── preprocessor.joblib         # Pipeline preprocessing salvata
│   ├── X_train.npz                 # Training set preprocessato
│   ├── X_test.npz                  # Test set preprocessato
│   ├── y_train.npy                 # Target training
│   ├── y_test.npy                  # Target test
│   ├── model_*.joblib              # Modelli ML classici salvati
│   ├── model_dl.keras              # Modello Deep Learning
│   ├── model_cat.cbm               # Modello CatBoost
│   ├── best_model_unified.joblib   # Ensemble wrapper unificato
│   ├── model_type.txt              # Tipo modello attivo
│   ├── threshold.txt               # Soglia decisionale ottimizzata
│   ├── confusion_matrix.png        # Matrice di confusione
│   └── *.png                       # Grafici EDA e SHAP
│
├── patients_data.db                # Database SQLite pazienti (generato)
│
└── [Script di utilità]
    ├── diagnose_dataset.py         # Analisi diagnostica dataset
    ├── seed_db.py                  # Popolamento database di test
    ├── analyze_data.py             # Analisi rapida dati
    └── check_corr.py               # Verifica correlazioni
```

---

## 🏗️ Architettura e Decisioni Progettuali

### 1. Separazione Moduli in Processi Isolati

**Decisione**: Ogni modulo (01, 02, 03, 04) viene eseguito come processo separato via `subprocess`.

**Motivazione**:
- **Isolamento memoria**: Evita memory leak accumulati tra TensorFlow, SHAP, matplotlib
- **Stabilità**: Prevenzione segmentation fault comuni su macOS/Anaconda
- **Debugging**: Più facile isolare errori specifici per modulo

**Implementazione**: `main.py` ⭢ `run_module()` ⭢ `subprocess.run()`

### 2. Ensemble di Modelli Multipli (8 modelli)

**Decisione**: Training di 8 modelli diversi: LR, RF, ET, GB, XGB, LGBM, CAT, DL

**Motivazione**:
- **Robustezza**: Diversi algoritmi catturano pattern diversi
- **Non-conoscenza a priori**: Non sappiamo quale funzioni meglio per questo dataset
- **Ridondanza**: Se un modello fallisce, gli altri compensano

**Combinazione**: Ensemble pesato con pesi esperti basati su performance attese:
- XGBoost, LightGBM, CatBoost: peso 4 (best-in-class per tabular data)
- Extra Trees: peso 3 (buono per pattern non-lineari)
- Random Forest, Gradient Boosting, Deep Learning: peso 2 (complementari)
- Logistic Regression: peso 0 (baseline, non usato nell'ensemble finale)

**I "Giocatori" dell'Ensemble**:
1. **Peso 4: I Top Player (XGBoost, LightGBM, CatBoost)**: Questi sono algoritmi basati su Gradient Boosting Decision Trees. Sono attualmente lo standard industriale per i dati tabellari (quelli tipo Excel).
   - **Perché peso 4?** Sono estremamente veloci e precisi nel trovare relazioni complesse nei dati. Sono considerati i più affidabili per questo tipo di task.
2. **Peso 3: Lo specialista (Extra Trees)**: Simile alla Random Forest ma più "aggressivo" e casuale nella scelta dei tagli dei dati.
   - **Perché peso 3?** Serve a catturare schemi (pattern) che i modelli di boosting potrebbero ignorare, riducendo il rischio che il super-modello faccia errori sistematici.
3. **Peso 2: I Complementari (Random Forest, Gradient Boosting, Deep Learning)**:
   - **Random Forest / Gradient Boosting**: Sono solidi, ma spesso meno performanti delle versioni "evolute" (come XGBoost).
   - **Deep Learning**: Sulle tabelle spesso fatica più degli alberi di decisione, ma può vedere relazioni "globali" che agli altri sfuggono.
   - **Perché peso 2?** Agiscono come "rete di sicurezza". Portano diversità all'insieme senza però dominare la decisione.
4. **Peso 0: La Baseline (Logistic Regression)**: La regressione logistica è il modello più semplice possibile.
   - **Perché peso 0?** Si usa come pietra di paragone. Se il super-modello da "peso 4" non batte nemmeno la regressione logistica, significa che c'è qualcosa che non va nel progetto.

### 3. Bilanciamento Classi: Undersampling + SMOTE

**Problema**: Dataset estremamente sbilanciato (91.2% healthy vs 8.8% disease, ratio 10.38:1)

**Decisione**: Strategia a due fasi:
1. **Undersampling**: Riduce classe maggioritaria da 10.38:1 a ~3:1
2. **SMOTE**: Oversampling sintetico per bilanciare a 1:1

**Motivazione**:
1. Se usassi solo lo SMOTE (che crea dati sintetici "copiando" i malati), rischierei di creare troppi dati artificiali basati su pochi esempi reali, portando il modello a imparare errori o rumore.
   - **SMOTE fallisce da solo**: Se il rapporto è troppo alto (10:1), lo SMOTE crea dati sintetici in zone "pericolose", sovrapponendoli ai sani e creando molta confusione (overfitting).
2. Se usassi solo l'Undersampling (cancellando i sani), perderei troppe informazioni utili.
   - **Rappresentatività**: L'undersampling moderato (fino a 3:1) assicura che i sani rimasti siano comunque un campione statisticamente valido di tutta la popolazione sana.
3. Trade-off Bias-Variance:
   - Il **Bias** (Distorsione) diminuisce perché il modello non è più "prevenuto" verso i sani.
   - La **Variance** (Variabilità) è sotto controllo perché non ho esagerato né nel cancellare dati reali né nell'inventarne di troppi sintetici.

### 4. Threshold Tournament

Il Torneo delle Soglie è la fase finale in cui decido "dove tirare la riga" per trasformare le probabilità del modello in una decisione netta: Sano o Malato.

Normalmente, un modello assegna un punteggio da 0 a 1 (es. 0.65). Di default, i computer usano 0.50 come soglia, ma in ambito medico questa soglia standard è quasi sempre sbagliata.

**Decisione**: Sweep completo delle soglie (0.01-0.99, step 0.01) e selezione basata su multiple metriche.
Invece di accettare il limite standard di 0.50, faccio una scansione: testo il modello 99 volte, spostando la soglia di 0.01 alla volta.
- Se abbasso la soglia (es. 0.10): Sarò molto severo, troverò tutti i malati ma avrò molti sani scambiati per malati (falsi positivi).
- Se alzi la soglia (es. 0.90): Sarò molto cauto, diagnosticherò solo i casi sicuri al 100%, ma perderò molti malati (falsi negativi).

**Metriche valutate**: Ogni metrica valuta la soglia da un punto di vista diverso.
- **F1-Score**: Bilancia precision e recall ⭢ Cerca l'equilibrio perfetto tra non sbagliare diagnosi e non dare falsi allarmi.
- **F2-Score**: Privilegia recall (importante in clinica) ⭢ È una versione "corretta" dell'F1. Dà più valore al Recall (trovare i malati). In medicina si preferisce perché è peggio ignorare un malato che preoccupare un sano inutilmente.
- **MCC (Matthews Correlation Coefficient)**: Correlazione predizioni-realtà ⭢ È considerato il "gold standard" per i dati sbilanciati. Dice quanto la previsione è meglio di una fatta a caso.
- **Youden's Index**: Sensitivity + Specificity - 1 ⭢ Cerca la soglia che massimizza la capacità del modello di distinguere i due gruppi.
- **G-Mean**: Geometric mean di sensitivity e specificity ⭢ Una media che "punisce" duramente il modello se ignora completamente una delle due classi (evita che il modello si concentri solo sui sani).
- **ROC Distance**: Distanza euclidea dal classificatore perfetto (1,1) ⭢ Immagina un grafico con un punto "perfetto" in alto a sinistra (100% di successo). Questa metrica misura quale soglia ci porta fisicamente più vicini a quel punto magico.

**Selezione finale**: Il "vincitore" del torneo non è scelto a caso, ma segue una gerarchia logica.
1. **Priorità Recall ≥ 0.70**: "Prima di tutto, devo trovare almeno il 70% dei malati reali". Qualsiasi soglia che ne trovi di meno viene squalificata subito, a prescindere da quanto sia precisa.
2. **Minimizzare i Falsi Positivi**: Tra le soglie che hanno passato il primo test (quelle che trovano abbastanza malati), scelgo quella che disturba meno persone sane.

### 5. Class Weights durante Training

Anche se ho già bilanciato i dati con Undersampling e SMOTE, i Class Weights sono un'ulteriore manopola per dire al modello: "Anche se ora i dati sono bilanciati come numero, non dare la stessa importanza agli errori".

**Decisione**: Class weights `{0: 1.0, 1: 0.90}` per penalizzare moderatamente la classe positiva.

**Coefficienti di penalità per gli errori del modello**:
- **Classe 0 (Sani - 1.0)**: Ogni volta che il modello scambia un sano per un malato, riceve una penalità "piena" (1.0).
- **Classe 1 (Malati - 0.90)**: Ogni volta che il modello non vede un malato (falso negativo), riceve una penalità leggermente ridotta (0.90).

**Perché penalizzare la classe positiva (i malati)?** Potrebbe sembrare controintuitivo in medicina, dove di solito si vuole trovare ogni singolo malato. Tuttavia, ci sono tre motivi tecnici fondamentali per questa scelta:
1. **Ridurre i Falsi Positivi**: Se lo SMOTE ha creato troppi dati sintetici "aggressivi", il modello potrebbe diventare paranoico e vedere malati ovunque. Abbassando il peso a 0.90, lo rendo un po' più cauto.
2. **Evitare l'Overfitting**: La classe 1 (i malati) era originariamente molto piccola (8.8%). Se la peso troppo, il modello potrebbe imparare a memoria quei pochi casi reali invece di capire la regola generale. Una penalizzazione leggera (0.90 invece di 1.0) evita che il modello si fissi troppo su dettagli insignificanti della classe minoritaria.
3. **Contesto Clinico**: In alcuni screening, un eccesso di falsi positivi può portare a esami invasivi, costosi e inutili per migliaia di persone. Questo peso cerca un punto di equilibrio per non "intasare" il sistema sanitario con falsi allarmi.

### 6. Preprocessing: StandardScaler + OneHotEncoder

**Decisione**: 
- Numeriche: `StandardScaler` (media 0, std 1)
- Categoriche: `OneHotEncoder` con `handle_unknown='ignore'`

**Nota**: Perché handle_unknown='ignore'? È una misura di sicurezza per l'Inference (quando il modello lavorerà nel mondo reale). Se domani arriva una categoria mai vista prima (es. "Colore_Giallo"), invece di andare in crash, il modello metterà semplicemente tutti zero, ignorando l'informazione sconosciuta ma continuando a funzionare.

**Motivazione**:
- **StandardScaler**: Modelli lineari e deep learning richiedono scale uniformi
- **OneHotEncoder**: Massima robustezza in inference (gestisce categorie nuove)
- **No OrdinalEncoder**: L'OrdinalEncoder trasforma le categorie in numeri progressivi (es. Rosso=1, Verde=2, Blu=3).
   - **Il rischio**: Il modello potrebbe pensare che Blu (3) sia "più grande" o "migliore" di Rosso (1) solo perché il numero è più alto.
   - **La scelta**: Usando il OneHotEncoder, accetto di avere più colonne (più memoria occupata) ma garantisco che il modello tratti ogni categoria come indipendente, senza inventare gerarchie che non esistono. Questo aiuta la generalizzazione, ovvero la capacità del modello di funzionare bene su dati nuovi.

### 7. Feature Engineering Conservativo

**Decisione**: Feature engineering minimale:
- `GeneralHealth` ⭢ `GeneralHealth_Num` (mapping ordinale ⭢ qui si è deciso che per la salute generale l'ordine conta)
   - **Logica**: "Pessimo" < "Sufficiente" < "Buono" < "Eccellente".
   - **Cosa fa**: Trasforma le parole in una scala numerica (es. 1, 2, 3, 4).
- `Sleep_Health_Ratio` = `SleepHours / (PhysicalHealthDays + 1)` (+1 per evitare la divisione per zero: se un paziente ha 0 giorni di malattia, la formula non crasha)
   - **Cosa rappresenta:** È un indicatore di "Resilienza".
      - Un valore alto indica una persona che dorme bene e ha pochi problemi fisici.
      - Un valore basso indica una persona che, nonostante il sonno (o per mancanza di esso), ha molti giorni di malessere.

**Motivazione**:
- Dataset già ricco di feature (31 colonne)
- Feature engineering estensivo rischia overfitting
- Focus su feature clinicamente interpretabili

### 8. Database SQLite per Storico Pazienti

**Decisione**: Tutte le predizioni salvate in `patients_data.db` con schema completo.

**Motivazione**:
- **Tracciabilità**: Storico completo per ogni paziente
- **Riapplicazione modelli**: Possibilità di ri-predire con modelli aggiornati
- **Analisi retrospettive**: Studio di pattern nel tempo

---

## 📦 Moduli del Sistema

### `main.py` - Dashboard CLI Principale

**Scopo**: Entry point del sistema, orchestratore centrale

**Funzioni**:
- Menu interattivo principale
- Routing verso moduli 01-04
- **Gestione modelli esistenti**: Verifica presenza modelli addestrati e offre opzione di skip training
- Gestione manutenzione (reset artifacts, delete database)
- Visualizzazione guida clinica

**Gestione Modelli Esistenti**:
- All'opzione `2` (Training), verifica automaticamente se esistono modelli già addestrati
- Se trovati, offre scelta tra:
  - Usare modelli esistenti (skip training)
  - Rifare training da zero (sovrascrive modelli)
- Permette di lavorare con modelli salvati senza rifare training ogni volta

**Dipendenze**: Tutti i moduli, Rich per UI
- **Rich**: Libreria Python progettata per rendere la CLI (l'interfaccia a riga di comando che abbiamo visto prima) visivamente accattivante e facile da leggere.
- **UI**: User Interface

---

### `notebooks/shared_utils.py` - Utility Condivise

**Scopo**: Configurazioni globali, funzioni utility, setup ambiente

**Contenuti**:
- Path definitions (DATASET_PATH, ARTIFACTS_DIR, DB_PATH)
- `setup_environment()`: Variabili d'ambiente per stabilità macOS
- `init_db()`: Inizializzazione schema database
- `get_db_connection()`: Context manager per connessioni DB
- `CLINICAL_GUIDE`: Glossario definizioni cliniche
- `check_models_exist()`: Verifica presenza modelli addestrati completi
- Console Rich configurata con tema custom
- `EnsembleWrapper`: Classe wrapper per ensemble modelli (usata per SHAP e inference)

**Funzione `check_models_exist()`**:
- Verifica presenza di tutti i file necessari per considerare il training completo
- Controlla: preprocessor, 7 modelli ML, modello DL, ensemble wrapper, threshold, model_type
- Usata da `main.py` per decidere se offrire opzione di skip training

**Note**: Modifiche qui impattano tutto il sistema

---

### `notebooks/01_EDA_Preprocessing.py` - EDA e Preprocessing

**Scopo**: Analisi esplorativa dati, feature engineering, preprocessing, salvataggio artifacts

**Processo**:
1. Caricamento dataset `heart_2022_no_nans.csv`
2. Creazione target `HeartDisease` (HadHeartAttack OR HadAngina)
3. Feature engineering (GeneralHealth_Num, Sleep_Health_Ratio)
4. Feature selection (rimozione SleepHours - correlazione ~0.009)
5. Separazione numeriche/categoriche
6. Preprocessing (StandardScaler + OneHotEncoder)
7. Train/Test split (80/20, stratificato): Lo split stratificato mantiene la stessa proporzione tra le classi (91.2% sani e 8.8% malati) in entrambi i set.
8. Salvataggio artifacts (.joblib, .npz, .npy)
   - **.joblib (Il Modello "Congelato")**: È il formato standard utilizzato dalla libreria scikit-learn per salvare i modelli di Machine Learning addestrati.
      - **Cosa contiene**: L'intera "intelligenza" del modello.
      - **Perché si usa**: Invece di ri-addestrare il modello ogni volta che riavvio il computer, carico il file .joblib e il modello è subito pronto per fare previsioni.
      - **Vantaggio**: È estremamente efficiente con oggetti Python che contengono grandi matrici di dati (tipico dei modelli AI).
   - **.npy (Matrici NumPy singole)**: Questa estensione appartiene a NumPy, la libreria fondamentale per il calcolo scientifico in Python.
      - **Cosa contiene**: Una singola "matrice" (array) di numeri (es. per salvare i valori SHAP calcolati per un gruppo di pazienti).
      - **Perché si usa**: È un formato binario molto veloce da leggere e scrivere, molto più compatto di un file Excel o CSV.
      - **Esempio**: Se calcolo l'importanza delle feature per 1000 pazienti, salvo il risultato in shap_values.npy.
   - **.npz (L'Archivio di Matrici)**: Pensa a questo come a un file "ZIP" specifico per i dati numerici di NumPy.
      - **Cosa contiene**: Diversi file .npy impacchettati insieme in un unico file.
      - **Perché si usa**: Serve a tenere i dati organizzati. Invece di avere 10 file diversi sul desktop, salvo tutto in un unico file .npz.

**Funzioni di visualizzazione**:
- Correlation Matrix (heatmap feature numeriche)
- Target Distribution (conteggio e percentuale)
- Feature vs Target (top 6 feature categoriche)
- Age & Health Analysis (età, salute generale, BMI, PhysicalHealthDays)
- Risk Factors (diabete, stroke, COPD, fumo, attività fisica)
- Top Correlations (15 feature più correlate con target)

**Argomenti command line**:
- `--no-plots`: Solo preprocessing, nessun grafico
- `--plot <tipo>`: Preprocessing + grafico specifico (correlation, target, features, age_health, risk_factors, top_correlations)

**Output artifacts**: `preprocessor.joblib`, `X_train.npz`, `X_test.npz`, `y_train.npy`, `y_test.npy`, `feature_names.joblib`

---

### `notebooks/02_Unified_Training.py` - Training Ensemble

**Scopo**: Training 8 modelli ML/DL, bilanciamento classi, ottimizzazione threshold, salvataggio modelli

**Processo**:
1. Caricamento dati preprocessati (da artifacts/)
2. **Bilanciamento classi**:
   - Undersampling classe maggioritaria (10.38:1 ⭢ 3:1)
   - SMOTE oversampling (3:1 ⭢ 1:1)
3. **Training 8 modelli** con class weights:
   - Logistic Regression
   - Random Forest
   - Extra Trees
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost
   - Neural Network (Keras)
4. **Predizioni ensemble pesate** (pesi esperti)
5. **Threshold Tournament**: Sweep soglie e selezione ottimale
6. **Valutazione finale**: Confusion matrix, metriche cliniche
7. **Salvataggio automatico** di tutti i modelli e artifacts finali in `artifacts/`

**Salvataggio Modelli**:
- Tutti i modelli vengono salvati automaticamente al termine del training
- Modelli ML classici: `.joblib` (LR, RF, ET, GB, XGB, LGBM)
- CatBoost: `.cbm` (formato proprietario CatBoost)
- Deep Learning: `.keras` (TensorFlow/Keras)
- Ensemble wrapper: `best_model_unified.joblib` (per SHAP e inference semplificata)
- Metadati: `model_type.txt` (tipo modello), `threshold.txt` (soglia ottimizzata)
- I modelli salvati possono essere riutilizzati in sessioni future (vedi `main.py`)

**Metriche calcolate**:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Sensitivity, Specificity, NPV, PPV
- MCC, Youden's Index, G-Mean

**Output artifacts**: `model_*.joblib`, `model_dl.keras`, `model_cat.cbm`, `best_model_unified.joblib`, `model_type.txt`, `threshold.txt`, `confusion_matrix.png`

---

### `notebooks/03_Explainability.py` - Explainability SHAP

**Scopo**: Interpretabilità modelli tramite SHAP (globale e locale)

L'analisi SHAP (SHapley Additive exPlanations) è un metodo utilizzato nel campo dell'intelligenza artificiale per spiegare l'output di modelli di machine learning. È considerato uno degli standard più avanzati per la cosiddetta XAI (Explainable AI), ovvero l'intelligenza artificiale spiegabile.
In parole semplici, SHAP ti dice "quanto e come" ogni singola informazione (variabile) ha influenzato la decisione finale del modello.

**Come funziona tecnicamente**: SHAP confronta la previsione del modello con e senza una determinata caratteristica. Tuttavia, non lo fa in isolamento; lo fa testando la caratteristica in tutte le possibili combinazioni con le altre variabili.

Il risultato è un valore SHAP per ogni variabile:
1. **Valore positivo**: La variabile ha spinto il modello verso un valore più alto (es. ha aumentato il rischio di malattia).
2. **Valore negativo**: La variabile ha spinto il modello verso un valore più basso.
3. **Valore vicino allo zero**: La variabile ha avuto poco impatto sulla decisione.

**Processo**: Questo punto descrive il "motore" che permette di capire cosa succede dentro la "black box" (la scatola nera) dell'Intelligenza Artificiale. È il processo che trasforma un freddo numero (es. "Rischio 85%") in una spiegazione clinica comprensibile.
1. Caricamento modello unificato (ensemble wrapper o singolo modello) ⭢ Viene caricato il modello (il file .joblib visto prima). Se è un Ensemble Wrapper, significa che il sistema carica più modelli insieme e li gestisce come se fossero uno solo.
2. Ricostruzione feature names dal preprocessor ⭢ Dopo il preprocessing (OneHotEncoder), le colonne originali cambiano nome e numero. Questa fase serve a mappare i risultati dell'IA ai nomi delle variabili umane (es. da col_0 a Fumatore_Sì).
3. **Feature Importance globale** (coefficienti o importances albero) ⭢ Prima di usare SHAP, si guarda la classifica "di fabbrica" del modello.
   - **Coefficienti**: Per i modelli lineari (es. Logistic Regression), ci dicono quanto pesa ogni variabile.
   - **Importances Albero**: Per modelli come XGBoost o Random Forest, indicano quante volte una variabile è stata usata per "dividere" i pazienti sani dai malati.
   - **Limite**: Questa analisi ti dice cosa è importante, ma non ti dice perché o in che direzione (es. "l'età conta", ma non dice se essere vecchi aumenta o diminuisce il rischio).
4. **SHAP Explainer** ⭢ SHAP sceglie uno strumento diverso in base al tipo di modello per essere il più efficiente possibile:
   - Ensemble: KernelExplainer ⭢ Si usa per l'Ensemble. Poiché l'ensemble è un mix di modelli diversi, SHAP lo tratta come una scatola nera e usa un metodo statistico (più lento) per stimare l'importanza delle variabili.
   - Tree models: TreeExplainer⭢ È un algoritmo ottimizzato specificamente per gli alberi (XGBoost, CatBoost). È velocissimo e matematicamente esatto.
   - Linear models: Explainer generico ⭢ Usato per modelli semplici (come la Logistic Regression).
5. **SHAP Summary Plot**: Questo grafico mostra l'importanza globale delle feature su tutti i pazienti.
   - Ogni riga è una variabile (es. Pressione Arteriosa).
   - Ogni punto è un paziente.
   - Il colore indica il valore (Rosso = Pressione Alta, Blu = Pressione Bassa).
   - Interpretazione: Se vedo punti rossi tutti a destra, capisco subito che "Pressione Alta" aumenta il rischio di malattia in tutta la mia popolazione.
6. **SHAP Local Plot**: Questa è la parte più utile per un medico durante una visita. Spiega una singola previsione specifica.
   - **Waterfall Plot**: Parte dal rischio medio e aggiunge o toglie "mattoni" (le variabili del paziente) fino ad arrivare al rischio finale.
   - **Bar Plot**: Una versione semplificata che mostra quali fattori hanno pesato di più per quel specifico individuo.
   - **Esempio pratico**: Il modello dice che il Paziente Rossi è a rischio (80%). Il Waterfall Plot mostra che il rischio è salito per via del Fumo (+30%) e della Sedentarietà (+20%), nonostante la Buona Dieta (-10%) abbia provato a compensare.

**Può essere rimosso?** ⚠️ **OPZIONALE** - Utile per interpretabilità, non necessario per predizioni

**Dipendenze**: SHAP, matplotlib, rich per output

---

### `notebooks/04_Inference.py` - Predizione Pazienti

**Scopo**: Interfaccia predizione pazienti (interattiva e batch), salvataggio database

**Funzioni principali**:
- `load_resources()`: Carica modelli addestrati, preprocessor e threshold ottimizzata
- `get_interactive_input()`: CLI interattiva per inserimento dati paziente
- `add_engineered_features()`: Feature engineering identico al training (GeneralHealth_Num, Sleep_Health_Ratio)
- `predict_ensemble()`: Predizione singolo paziente via ensemble pesato
- `predict_batch()`: Predizione batch per tutti i pazienti nel database
- `save_to_db()`: Salvataggio predizioni su SQLite con gestione schema automatica

**Processo predizione interattiva**:
1. Raccolta dati clinici tramite CLI (Command Line Interface) interattiva
2. Feature engineering (coerente con Module 01)
3. Preprocessing (StandardScaler + OneHotEncoder)
4. Predizioni da 8 modelli ML/DL
5. Combinazione pesata delle probabilità (pesi esperti)
6. Decisione binaria basata su threshold ottimizzata
7. Visualizzazione risultato e **salvataggio automatico nel database**

**Gestione database**:
- `save_to_db()` gestisce automaticamente:
  - **Rimozione colonne feature engineering** (GeneralHealth_Num, Sleep_Health_Ratio) prima del salvataggio
  - **Aggiunta colonne mancanti** con valori di default per compatibilità schema
  - **Salvataggio sicuro** con gestione errori esplicita
- Pazienti salvati disponibili per batch prediction (opzione 5)

**Argomenti command line**:
- Nessuno: Predizione interattiva singolo paziente (con salvataggio automatico DB)
- `--batch`: Predizione batch tutti pazienti database (aggiorna predizioni esistenti)

**Note importanti**:
- Feature engineering DEVE essere identico a Module 01 per coerenza
- I pazienti vengono salvati automaticamente dopo ogni predizione interattiva
- Le colonne di feature engineering vengono rimosse prima del salvataggio (non fanno parte dello schema DB)

---

### Script di Utilità

#### `diagnose_dataset.py`

**Scopo**: Analisi diagnostica dataset (correlazioni feature-target, feature deboli, variabilità)

**Quando usare**: Per capire qualità dataset prima di training

**Può essere rimosso?** ✅ **SÌ** - Solo diagnostica, non necessario per funzionamento

#### `seed_db.py`

**Scopo**: Popola database con record di test per testing

**Può essere rimosso?** ✅ **SÌ** - Solo per development/testing

#### `analyze_data.py`, `check_corr.py`

**Scopo**: Script di analisi rapida dataset

**Può essere rimosso?** ✅ **SÌ** - Utility opzionali

---

## 📊 Dataset e Limitazioni

### Dataset CDC 2022: Indicators of Heart Disease (2022 UPDATE)

**Fonte**: CDC (Centers for Disease Control and Prevention) 2022 BRFSS (Behavioral Risk Factor Surveillance System) [link Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

**Dimensioni**: 246,022 record, 40 colonne originali

**Target**: `HeartDisease` = 1 se `HadHeartAttack == 'Yes'` OR `HadAngina == 'Yes'`

**Distribuzione**:
- **Classe 0 (Healthy)**: 224 406 (91.2%)
- **Classe 1 (Disease)**: 21 616 (8.8%)
- **Imbalance Ratio**: 10.38:1

### Problemi Identificati del Dataset

#### 1. **Class Imbalance Estremo**

**Problema**: 91.2% vs 8.8% crea bias verso classe maggioritaria

**Impatto**:
- Precision bassa (~25%): molti falsi positivi
- Modello conservativo: preferisce predire "healthy"
- Performance limitata anche con SMOTE

**Soluzione implementata**: Undersampling + SMOTE (vedi sezione architettura)

**Risultati attesi**:
- Precision: ~25% ⭢ ~30-35%
- Recall: ~75% ⭢ mantenuta ~70-75%
- Falsi positivi: ~10,000 ⭢ ~8,000-9,000

#### 2. **Correlazioni Feature-Target Basse**

**Nota**: r sta per Coefficiente di Correlazione di Pearson (r di Pearson)
- È un valore numerico che misura quanto due variabili siano legate tra loro da un rapporto di tipo lineare. In questo caso, indica quanto forte sia il legame tra ogni caratteristica (feature) e la malattia (target).
- Il valore di r oscilla sempre tra -1 e +1:
   - r = +1: Correlazione positiva perfetta (se una aumenta, l'altra aumenta proporzionalmente).
   - r = 0: Nessuna correlazione (le variabili sono indipendenti, come il colore dei tuoi calzini e il rischio di pioggia).
   - r = −1: Correlazione negativa perfetta (se una aumenta, l'altra diminuisce).

**Analisi** (`diagnose_dataset.py`):
- Top correlazione: `GeneralHealth` (r = 0.243) ⭢ Al variare della percezione della salute (es. da "ottima" a "pessima" nel mapping numerico) aumenta la probabilità di avere la malattia. Un valore di 0.24 è considerato una correlazione debole-moderata, ma in medicina è spesso un segnale molto significativo.
- Solo 16 feature con r > 0.1 ⭢ Sono le variabili che hanno un "peso" visibile. Sono quelle su cui il modello farà più affidamento.
- 8 feature con r < 0.05 (deboli) ⭢ Queste variabili sono quasi "rumore". Da sole non spiegano quasi nulla della malattia.

**Impatto**:
- Feature non molto predittive limitano performance massime
- Modello fatica a distinguere pattern sottili
- Precision probabilmente non supererà ~35% anche con miglioramenti

**Spiegazione**:
- Dataset basato su survey self-reported (non misurazioni oggettive)
- Manca informazione diretta (pressione arteriosa, colesterolo HDL/LDL, ECG)
- Comorbidità e sintomi sono correlati ma non diagnostici
   - In ambito medico, la comorbidità (o comorbilità) è la presenza contemporanea di due o più malattie o disturbi diversi nello stesso paziente.
   - **Correlati**: Significa che spesso queste malattie "viaggiano insieme". Chi ha una, ha più probabilità di avere anche l'altra.
   - **Non diagnostici**: Significa che la sola presenza della comorbidità non è una prova certa della malattia principale che stai cercando di prevedere.
   - **Esempio**: Il mio modello cerca di prevedere una specifica malattia cardiaca, il fatto che il paziente abbia il "diabete" (comorbidità) è un segnale importante, ma non basta a dire "Sì, ha sicuramente quella malattia cardiaca".
   - Giustifica l'uso di tecniche avanzate come l'Ensemble e lo SHAP:
      - **Segnali deboli**: Se le comorbidità non sono diagnostiche, significa che il modello non troverà mai una "risposta facile". Deve invece imparare a combinare tanti piccoli segnali (ipertensione + età + fumo + sintomi) per arrivare a una conclusione affidabile.
      - **Rischio di confusione**: Il modello potrebbe confondersi e pensare che la comorbidità sia la causa della malattia, mentre è solo una condizione che spesso la accompagna.
      - **Utilità dello SHAP**: Qui SHAP diventa fondamentale. Permette di vedere se il modello ha deciso "Malato" perché ha visto la malattia vera o se si è fatto "ingannare" dalla presenza di una comorbidità molto comune.

**Soluzione futura**: 
- Integrare dati oggettivi (lab values, ECG)
- Feature engineering avanzato (interazioni cliniche)
- Dataset esterni arricchiti

#### 3. **Feature SleepHours Quasi Inutile**

**Problema**: Correlazione ~0.009 con target (quasi zero)

**Soluzione**: Rimossa in preprocessing, mantenuto `Sleep_Health_Ratio` (r = 0.115)

### Limitazioni Attuali del Sistema

1. **Performance limitata da dataset**: Precision ~30-35% è realistica data qualità feature
   - **La Precision risponde alla domanda**: "Se il modello dice che il paziente è malato, quanto è probabile che lo sia davvero?".
   - **Significato**: Solo 1 persona su 3 identificate dal modello ha realmente la malattia.
2. **Recall clinica**: 70-75% significa ~25-30% pazienti con malattia non identificati (critico in clinica)
   - **La Recall risponde alla domanda**: "Su 100 malati veri, quanti ne ha trovati il modello?".
   - **Il problema**: Se la Recall è al 70%, significa che il 30% dei malati riceve una diagnosi errata di "Sano".
   - **Impatto Clinico**: In medicina, questo è l'errore più pericoloso (Falso Negativo). Significa che 3 pazienti su 10 che avrebbero bisogno di cure vengono rimandati a casa.
3. **Falsi positivi alti**: ~9,000 FP su ~49,000 test cases (18% falsi allarmi)
   - **Conseguenza**: Il 18% dei pazienti sani verrebbe sottoposto a esami di secondo livello (più costosi o invasivi) inutilmente.
   - **Carico di lavoro**: Per un ospedale, gestire 9.000 "sospetti" per trovarne poi solo una frazione realmente malata potrebbe essere insostenibile a livello di costi e personale.
4. **Nessuna calibrazione probabilità**: Probabilità non calibrate, usare solo per ranking
   - **Il problema**: Se il modello dice "Rischio 80%", in un modello non calibrato quel numero non significa che il paziente ha l'80% di probabilità reale di essere malato. Potrebbe essere solo un punteggio interno.
   - **Cosa fare (Ranking)**: Va usato il valore solo per fare una classifica. Ovvero: un paziente con 0.90 è quasi certamente "più a rischio" di uno con 0.70, ma non sappiamo l'esatta probabilità medica.

### Raccomandazioni per Uso Clinico

⚠️ **NON USARE PER DIAGNOSI DEFINITIVA**

**Uso appropriato**:
- **Screening iniziale**: Identificare pazienti ad alto rischio per screening approfondito
- **Prioritizzazione**: Assegnare priorità a pazienti per visita specialistica
- **Supporto decisionale**: Combinare con giudizio clinico, non sostituire

**Limitazioni da comunicare**:
- Precision ~30-35%: molti falsi positivi
- Recall ~70-75%: alcuni casi reali mancati
- Basato su survey self-reported, non misurazioni oggettive

---

## ⚙️ Installazione e Configurazione

### Requisiti di Sistema

- **Python**: 3.8+
- **OS**: macOS, Linux, Windows
- **RAM**: Minimo 4GB (8GB consigliato per training)
- **Disk**: ~2GB per dataset e artifacts

### Installazione Dipendenze

```bash
pip install -r requirements.txt
```

**Dipendenze principali**:
- `pandas`, `numpy`: Manipolazione dati
- `scikit-learn`: Machine Learning classico
- `xgboost`, `lightgbm`, `catboost`: Gradient boosting
- `tensorflow`: Deep Learning
- `shap`: Explainability
- `rich`: CLI colorata
- `matplotlib`, `seaborn`: Visualizzazioni
- `joblib`: Serializzazione modelli
- `imblearn`: Bilanciamento classi (SMOTE)

### Configurazione Environment

Il sistema configura automaticamente variabili d'ambiente per stabilità macOS:

```python
KMP_DUPLICATE_LIB_OK=True      # Evita conflitti OpenMP
OMP_NUM_THREADS=1              # Stabilità TensorFlow
MKL_NUM_THREADS=1              # Stabilità NumPy
TF_CPP_MIN_LOG_LEVEL=2         # Riduce log TensorFlow
TF_METAL_DEVICE_NS=0           # Disabilita Metal (stabilità)
```

**Note macOS**: Se usi Anaconda, assicurati di NON essere nell'ambiente `base`. Usa un ambiente virtuale dedicato.

---

## 🚀 Utilizzo del Sistema

### Avvio Dashboard

```bash
python main.py
```

### Flusso Operativo Completo

#### **STEP 1: Preprocessing e EDA**

Menu principale ⭢ Opzione `1` ⭢ EDA & Visual Analysis

**Cosa fa**:
1. Esegue automaticamente preprocessing (feature engineering, scaling, encoding)
2. Salva artifacts in `artifacts/`
3. Apre sottomenu visualizzazioni

**Sottomenu EDA**:
- `1`: Correlation Matrix (heatmap)
- `2`: Target Distribution (distribuzione classi)
- `3`: Feature vs Target (grafici feature per classe)
- `4`: Age & Health Analysis (analisi età/salute)
- `5`: Risk Factors (fattori di rischio)
- `6`: Top Correlations (15 feature più correlate)

**Output**: Artifacts salvati, grafici in `artifacts/*.png`

**Tempo**: ~1-2 minuti (dataset 246K record)

#### **STEP 2: Training Modelli**

Menu principale ⭢ Opzione `2` ⭢ Unified Model Competition

**Cosa fa**:
1. **Verifica modelli esistenti**: Controlla se esistono modelli già addestrati da una sessione precedente
2. **Menu di scelta** (se modelli esistono):
   - `use`: Usa modelli esistenti (skip training - risparmia tempo)
   - `retrain`: Rifai il training da zero (sovrascrive modelli esistenti)
3. **Training** (solo se richiesto o se nessun modello esistente):
   - Carica dati preprocessati
   - Bilanciamento classi (Undersampling + SMOTE)
   - Training 8 modelli ML/DL
   - Threshold Tournament (ottimizzazione soglia)
   - Valutazione finale con confusion matrix
   - **Salvataggio automatico** di tutti i modelli in `artifacts/`

**Sistema di Persistenza**:
- I modelli vengono **salvati automaticamente** dopo ogni training in `artifacts/`
- Alla prossima sessione, il sistema rileva automaticamente i modelli esistenti
- Puoi usare i modelli esistenti senza rifare il training (utile per evitare 10-30 minuti di attesa)
- I modelli rimangono disponibili finché non li elimini manualmente (opzione `r` - Reset Artifacts)

**Output**: 
- Modelli in `artifacts/model_*.joblib`, `model_dl.keras`, `model_cat.cbm`
- `best_model_unified.joblib`: Ensemble wrapper
- `threshold.txt`: Soglia ottimizzata
- `confusion_matrix.png`: Matrice confusione

**Tempo**: 
- Usando modelli esistenti: ~1 secondo (skip training)
- Nuovo training: ~10-30 minuti (dipende CPU/RAM)

**Metriche mostrate** (solo durante training):
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix con dettagli (TN, FP, FN, TP)
- Metriche cliniche (Sensitivity, Specificity, NPV, PPV)

#### **STEP 3: Explainability (Opzionale)**

Menu principale ⭢ Opzione `3` ⭢ Explainability (SHAP Analysis)

**Cosa fa**:
1. Carica modello unificato
2. Feature Importance globale
3. SHAP Summary Plot (importanza feature)
4. SHAP Local Plot (spiegazione singola predizione)

**Output**: Grafici SHAP in `artifacts/shap_*.png`

**Tempo**: ~2-5 minuti (SHAP è computazionalmente costoso)

#### **STEP 4: Predizione Pazienti**

**4a. Predizione Interattiva Singolo Paziente**

Menu principale ⭢ Opzione `4` ⭢ Predict for a New Patient

**Cosa fa**:
1. CLI interattiva per inserimento dati clinici
2. Feature engineering (coerente con training)
3. Predizione via ensemble pesato (8 modelli combinati)
4. **Salvataggio automatico nel database** con gestione schema automatica

**Input richiesto**: Tutti i campi clinici (demografia, salute, abitudini, condizioni mediche)

**Output**: 
- Pannello con predizione (High Risk / Low Risk), confidence, threshold
- **Messaggio di conferma salvataggio** nel database
- Paziente salvato disponibile per batch prediction (opzione 5)

**Gestione database automatica**:
- Rimuove automaticamente colonne di feature engineering (non nello schema DB)
- Aggiunge colonne mancanti con valori di default
- Gestione errori esplicita con messaggi informativi

**Tempo**: ~10-30 secondi (incluso inserimento dati e salvataggio)

**4b. Predizione Batch Tutti i Pazienti**

Menu principale ⭢ Opzione `5` ⭢ Batch Predict all Patients in Database

**Cosa fa**:
1. Carica tutti i pazienti dal database (aggiunti tramite opzione 4)
2. Feature engineering e preprocessing batch
3. Predizione ensemble per tutti i pazienti simultaneamente
4. Aggiornamento database con nuove predizioni e probabilità
5. Statistiche finali (totale, high risk, low risk, percentuali)

**Output**: 
- Statistiche complete (totale, high risk, low risk, percentuali)
- Database aggiornato con predizioni per tutti i pazienti

**Tempo**: Dipende da numero pazienti nel database (~1-2 secondi per paziente)

**Nota**: Se il database è vuoto, usa prima l'opzione 4 per aggiungere pazienti

#### **STEP 5: Guida Clinica (Opzionale)**

Menu principale ⭢ Opzione `6` ⭢ Clinical Glossary

**Cosa fa**: Visualizza definizioni e spiegazioni di tutti i parametri clinici

---

## 🔧 Manutenzione

### Reset Artifacts

Menu principale ⭢ Opzione `r` ⭢ Reset ML Artifacts

**Cosa fa**: Elimina cartella `artifacts/` (modelli, dati preprocessati, grafici)

**Quando usare**:
- Cambio Python environment
- Riexecuzione preprocessing con modifiche
- Pulizia spazio disco

**⚠️ Attenzione**: Dovrai rieseguire STEP 1 e STEP 2 per rigenerare artifacts

### Reset Database

Menu principale ⭢ Opzione `d` ⭢ Delete SQL Patient Database

**Cosa fa**: Elimina `patients_data.db` (storico pazienti, predizioni)

**Quando usare**:
- Test pulito
- Privacy (GDPR compliance): la conformità al Regolamento Generale sulla Protezione dei Dati non è solo un dettaglio tecnico, ma un obbligo legale severissimo in Europa.
- Reset storico predizioni

**⚠️ Attenzione**: Perdita completa storico pazienti

### Verifica Integrità Sistema

Controlla che esistano artifacts necessari:

```bash
# Controlla artifacts essenziali
ls artifacts/preprocessor.joblib
ls artifacts/best_model_unified.joblib
ls artifacts/threshold.txt
ls artifacts/model_type.txt
```

Se mancano, riesegui STEP 1 e STEP 2.

### Gestione Modelli Salvati

**Come funziona**:
- Dopo ogni training, tutti i modelli vengono salvati automaticamente in `artifacts/`
- Quando riavvii il programma e selezioni opzione `2` (Training), il sistema:
  - Rileva automaticamente se esistono modelli già addestrati
  - Ti offre la scelta di usare i modelli esistenti o rifare il training
- I modelli rimangono disponibili finché non li elimini manualmente (opzione `r`)

**Vantaggi**:
- **Risparmio tempo**: Non devi rifare training ogni volta (10-30 minuti risparmiati)
- **Continuità lavoro**: Continui a lavorare con i modelli dalla sessione precedente
- **Flessibilità**: Puoi sempre scegliere di rifare training se necessario

**Nota**: I modelli vengono sovrascritti solo se scegli esplicitamente `retrain` o se elimini gli artifacts (opzione `r`)

---

## 📈 Risultati Attesi e Performance

### Performance Attuali del Sistema

**Dati test (49 205 casi, 20% del dataset)**:

| Metrica | Valore | Interpretazione |
|---------|--------|-----------------|
| **Accuracy** | ~77-78% | Percentuale predizioni corrette |
| **Precision** | ~30-35% | Quando predice "disease", corretto ~30-35% volte |
| **Recall (Sensitivity)** | ~70-75% | Identifica ~70-75% pazienti con malattia reale |
| **Specificity** | ~77-80% | Identifica ~77-80% pazienti sani correttamente |
| **F1-Score** | ~45-50% | Media armonica precision-recall |
| **ROC-AUC** | ~0.80-0.85 | Area sotto curva ROC (buona discriminazione) |

**Confusion Matrix tipica**:
```
True Negatives (TN):  ~35,000  |  Predicted Healthy correctly
False Positives (FP):  ~9,500  |  Predicted Disease but actually Healthy
False Negatives (FN):  ~1,100  |  Predicted Healthy but actually Disease
True Positives (TP):  ~3,200   |  Predicted Disease correctly
```

### Limitazioni Performance

1. **Precision bassa (~30%)**: 9,500 falsi positivi significa 70% delle predizioni "disease" sono sbagliate
2. **Recall non ottimale (~75%)**: ~25% pazienti con malattia non identificati (critico in clinica)
3. **Dataset sbilanciato**: Limitazione intrinseca, difficilmente superabile
4. **Feature poco predittive**: Correlazioni max 0.24 limitano performance massime

### Interpretazione Clinica

**Cosa significa Precision ~30%?**
- Su 100 predizioni "High Risk", solo ~30 hanno realmente malattia
- **Implicazione**: Usare come screening, non diagnosi. Pazienti "High Risk" necessitano follow-up approfondito.

**Cosa significa Recall ~75%?**
- Su 100 pazienti con malattia reale, ~75 sono identificati, ~25 sono persi
- **Implicazione**: Non sostituisce screening clinico standard. Alcuni casi reali saranno mancati.

**Raccomandazione uso**:
- **Screening triage**: Prioritizzare pazienti "High Risk" per visita specialistica
- **Supporto decisionale**: Combinare con giudizio clinico, anamnesi, esami fisici
- **NON sostituzione**: Non usare come unico criterio diagnostico

---

## 🗺️ Roadmap e Miglioramenti Futuri

### Miglioramenti Breve Termine

- [ ] **Calibrazione probabilità**: Platt scaling o isotonic regression per probabilità calibrate
- [ ] **Feature selection automatica**: Rimozione feature con correlazione < 0.01 durante training
- [ ] **Cross-validation**: Aggiungere K-fold CV per metriche più robuste
- [ ] **Export predizioni**: Export CSV/Excel delle predizioni batch

### Miglioramenti Medio Termine

- [ ] **Integrazione dati oggettivi**: Pressione arteriosa, colesterolo HDL/LDL, ECG parameters
- [ ] **Feature engineering avanzato**: Interazioni cliniche, polynomial features
- [ ] **Modelli ensemble alternativi**: Stacking, Blending, Voting
- [ ] **Dashboard web**: Interfaccia web invece di CLI
- [ ] **API REST**: Endpoint HTTP per integrazione con sistemi esterni

### Miglioramenti Lungo Termine

- [ ] **Integrazione EHR**: Connessione con Electronic Health Record standards (HL7 FHIR)
- [ ] **Temporal modeling**: Modelli time-series per storico paziente
- [ ] **Federated learning**: Training distribuito su multiple istituzioni
- [ ] **Explainability avanzata**: LIME, counterfactual explanations
- [ ] **Deployment production**: Docker, Kubernetes, CI/CD

### Priorità Critiche per Uso Clinico

1. **Migliorare Recall**: Target ≥ 85% (ridurre falsi negativi)
2. **Calibrare probabilità**: Per ranking affidabile pazienti
3. **Integrare dati oggettivi**: Pressione, colesterolo, ECG per feature più predittive
4. **Validazione esterna**: Test su dataset esterni per generalizzazione

---

## 📚 Riferimenti e Bibliografia

### Dataset

- **CDC BRFSS 2022**: Behavioral Risk Factor Surveillance System
- Disponibile su Kaggle: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

### Metodologie ML

- **SMOTE**: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- **SHAP**: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- **Ensemble Methods**: Zhou (2012) "Ensemble Methods: Foundations and Algorithms"

### Medical ML

- **Clinical Decision Support**: Greenes (2014) "Clinical Decision Support Systems"
- **Explainable AI in Healthcare**: Arrieta et al. (2020) "Explainable Artificial Intelligence"

---

## 📝 Licenza e Note Legali

⚠️ **DISCLAIMER MEDICO**

Questo sistema è un **supporto decisionale**, non uno strumento diagnostico. Le predizioni non sostituiscono giudizio clinico, anamnesi, esami fisici o test diagnostici. Non usare per diagnosi definitiva o trattamento medico. L'uso è a proprio rischio.

**Limiti di responsabilità**: Lo sviluppatore non si assume responsabilità per decisioni cliniche basate su questo sistema.

---

## 👥 Autore: Emanuele Anzellotti

Progetto sviluppato per dimostrare applicazione Machine Learning in ambito clinico.

---

**Ultimo aggiornamento**: gennaio 2026
**Status**: Prototipo funzionante con:
- Sistema di persistenza modelli (salvataggio/caricamento automatico)
- Salvataggio database pazienti con gestione schema automatica
- Commenti dettagliati su tutto il codice
- Gestione errori robusta per salvataggio database
