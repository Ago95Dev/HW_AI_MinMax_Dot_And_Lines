# ISTRUZIONI PER L'ESECUZIONE

## 📦 Setup Iniziale

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

## 🎮 Esecuzione Rapida

### Test Componenti Singoli

```bash
# Dimostra il gioco
python dots_and_boxes.py

# Testa l'MLP
python mlp_evaluator.py

# Testa MinMax
python minmax.py

# Testa il training loop
python train_loop.py

# Visualizza le strategie
python adaptive_strategy.py
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

### Metodo 2: Notebook Headless

```bash
# Converte notebook in Python e esegue
jupyter nbconvert --to python experiment.ipynb
python experiment.py
```

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

## 📝 Generare il Report PDF

### Opzione 1: Da Notebook

```bash
# Avvia Jupyter e esporta
jupyter notebook experiment.ipynb
# File → Download as → PDF via LaTeX
```

### Opzione 2: Da Template LaTeX

```bash
# Compila il template LaTeX
./compile_report.sh

# Oppure manualmente
pdflatex report.tex
pdflatex report.tex  # Due volte per riferimenti
```

**Nota**: Devi compilare `report.tex` DOPO aver eseguito gli esperimenti per inserire i risultati effettivi nelle sezioni [TBD].

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

# 5. Consegna
# → experiment.ipynb (con output)
# → report.pdf
# → Codice sorgente (già presente)
# → Chat con AI assistant
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

### Notebook troppo lento
- Usa griglia 2×2 invece di 3×3
- Riduci GAMES_PER_ITERATION
- Seleziona meno strategie da testare

### LaTeX compilation error
- Installa texlive-full: `sudo apt-get install texlive-full`
- Oppure usa export PDF direttamente da Jupyter

## 📋 Checklist Consegna

- [ ] Codice sorgente completo
- [ ] `experiment.ipynb` con output dei risultati
- [ ] `report.pdf` compilato con risultati
- [ ] Chat/conversazione con AI assistant
- [ ] README.md con istruzioni

## 💡 Suggerimenti

1. **Primo run**: Usa configurazione veloce (grid_size=2, poche iterazioni) per verificare che tutto funzioni

2. **Esperimenti finali**: Aumenta parametri per risultati migliori
   - Grid size: 3×3
   - Iterations: 40-50
   - Games: 5-10 per iteration

3. **Grafici**: Tutti i grafici vengono salvati automaticamente come PNG. Includili nel report LaTeX o in una presentazione.

4. **Analisi**: Concentrati sulla strategia che ottiene la loss finale più bassa

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
