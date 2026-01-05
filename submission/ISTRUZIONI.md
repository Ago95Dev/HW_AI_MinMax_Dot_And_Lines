# ISTRUZIONI PER L'ESECUZIONE

##  Setup Iniziale

### 1. Installare le Dipendenze

Il progetto richiede Python 3.8+ e le seguenti librerie:

```bash
# Opzione A: Usando pip
pip install torch numpy matplotlib seaborn pandas jupyter tqdm

# Opzione B: Usando il file requirements.txt
pip install -r requirements.txt

# Opzione C: Usando conda
conda install pytorch numpy matplotlib seaborn pandas jupyter tqdm -c pytorch
```

### 2. Verificare l'Installazione

```bash
python test_all.py
```

Se tutto funziona, vedrai:
```
✓ All external libraries imported successfully
✓ Game engine working
✓ MLP evaluator working
✓ MinMax agent working
✓ Training loop working
✓ Adaptive strategies working

ALL TESTS PASSED! ✓
```

## 🎮 Esecuzione GUI (Gioco)
Per avviare l'interfaccia grafica e giocare:

```bash
# Metodo Rapido (Linux/Mac)
./run_game.sh

# Oppure con Python
python gui.py
```

## 🔬 Eseguire gli Esperimenti

### Metodo 1: Jupyter Notebook (Raccomandato)

```bash
# Avvia Jupyter
jupyter notebook experiment.ipynb

# Oppure usa JupyterLab
jupyter lab experiment.ipynb
```

Nel notebook:
1. Esegui tutte le celle in ordine (Cell → Run All)
2. Aspetta il completamento (~10-15 minuti per setup completo)
3. I grafici vengono salvati automaticamente come PNG

### Metodo 2: Script Python (Headless)
Se non vuoi usare Jupyter, puoi eseguire lo script Python che replica gli esperimenti e salva i grafici:

```bash
python run_experiments.py
```
Questo script:
1. Esegue il training delle strategie
2. Genera i grafici (`strategies_plot.png`, `training_loss_comparison.png`)
3. Esegue la "Generational Battle" e stampa i risultati
3. Esegue la "Generational Battle" e stampa i risultati
4. Esegue l'analisi Costi/Benefici

### Metodo 3: Nuovi Esperimenti (Strategie Avanzate)
Per testare le nuove strategie (`CuriosityDriven`, `PrecisionFirst`) e i benchmark:

```bash
python run_new_experiments.py
```
Questo script:
1. Allena le nuove strategie.
2. Confronta l'agente con `GreedyAgent` e `ClassicalMinMaxAgent`.
3. Genera grafici specifici per la stabilità del segnale e le performance contro i benchmark.

## 📊 Parametri Modificabili

Nel notebook `experiment.ipynb`, puoi modificare:

```python
# Sezione 3: Training Experiments

GRID_SIZE = 3              # 2, 3, o 4 (più grande = più lento)
NUM_ITERATIONS = 40        # Numero iterazioni training
GAMES_PER_ITERATION = 5    # Partite per iterazione
EPOCHS_PER_BATCH = 2       # Epoche training MLP

# Per test veloce:
# GRID_SIZE = 2
# NUM_ITERATIONS = 10
# GAMES_PER_ITERATION = 3
```

## 🎯 Workflow Completo

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Test
python test_all.py

# 3. Esperimenti
jupyter notebook experiment.ipynb
# → Esegui tutte le celle
# → Salva i grafici generati

# 4. Report
# → Copia risultati dal notebook a report.tex
# → Compila: ./compile_report.sh

```

## 🐛 Risoluzione Problemi

### "No module named 'torch'"
```bash
pip install torch
```

### "Jupyter not found"
```bash
pip install jupyter notebook
```

### "Out of memory" durante training
Riduci i parametri:
```python
GRID_SIZE = 2
NUM_ITERATIONS = 20
hidden_sizes = [64, 32]  # invece di [128, 64]
```

## 🚀 Quick Start (1 minuto)

```bash
# Clone/naviga nella directory
cd dot_and_lines_HW1_Caianiello

# Setup (una volta sola)
pip install -r requirements.txt

# Test veloce
python test_all.py

# Esperimenti completi
jupyter notebook experiment.ipynb
```

Fatto! 🎉
