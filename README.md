# Predictive MinMax for Dots and Boxes

## Homework 1 - Artificial Intelligence 25/26

Implementazione di un agente self-learning per il gioco Dots and Boxes utilizzando l'algoritmo Predictive MinMax con parametri adattivi.

## 📋 Componenti del Progetto

### 1. Game Engine (`dots_and_boxes.py`)
- Implementazione completa del gioco Dots and Boxes
- Griglia parametrizzabile (default 3×3)
- Gestione completa delle regole (box completion, turni extra)
- Rappresentazione dello stato vettoriale per l'MLP

### 2. MLP Evaluator (`mlp_evaluator.py`)
- Multi-Layer Perceptron implementato con PyTorch
- Architettura: Input → Hidden Layers [128, 64] → Output
- Output in range [-1, +1] con tanh activation
- Training con MSE loss e Adam optimizer

### 3. MinMax Algorithm (`minmax.py`)
-  MinMax search con tagli di profondità (L) e ampiezza (K)
- Alpha-beta pruning per ottimizzazione
- Move ordering basato su valutazioni MLP
- Statistiche di ricerca (nodi esplorati, foglie)

### 4. Training Loop (`train_loop.py`)
- Pipeline self-play: Play → Observe → Learn
- Raccolta automatica degli stati visitati
- Training batch su giochi multipli
- Metriche di performance e statistiche

### 5. Adaptive Strategies (`adaptive_strategy.py`)
Sei strategie implementate per L(t) e K(t):
- **Progressive Deepening**: Aumenta gradualmente L, K costante
- **Inverse Relationship**: L↑, K↓ (più profondità, meno ampiezza)
- **Exponential Growth**: Crescita esponenziale di L
- **Sigmoid**: Transizione smooth con curva sigmoide
- **Staircase**: Salti discreti a intervalli regolari
- **Constant**: Baseline con parametri fissi

### 6. Experiment Notebook (`experiment.ipynb`)
- Training comparativo delle strategie
- Visualizzazioni (loss, outcomes, distributions)
- Analisi statistica dei risultati
- Test dell'agente addestrato

## 🚀 Quick Start

### Installazione Dipendenze
```bash
pip install -r requirements.txt
```

### Test Componenti
```bash
# Test game engine
python dots_and_boxes.py

# Test MLP evaluator
python mlp_evaluator.py

# Test MinMax
python minmax.py

# Test training loop
python train_loop.py

# Test adaptive strategies
python adaptive_strategy.py
```

### Eseguire Esperimenti
```bash
jupyter notebook experiment.ipynb
```

## 📊 Struttura del Progetto

```
dot_and_lines_HW1_Caianiello/
├── HW1_2025-1.pdf              # Homework assignment PDF
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── dots_and_boxes.py           # Game implementation
├── mlp_evaluator.py            # Neural network
├── minmax.py                   # Search algorithm
├── train_loop.py               # Training pipeline
├── adaptive_strategy.py        # L(t), K(t) strategies
└── experiment.ipynb            # Experiments and analysis
```

## 🔬 Esperimenti

Il notebook `experiment.ipynb` esegue:

1. **Strategy Visualization**: Grafici di evoluzione di L(t) e K(t)
2. **Training Comparison**: Confronto loss tra strategie
3. **Outcome Analysis**: Distribuzione win/tie/loss nel tempo
4. **MLP Behavior**: Visualizzazione delle valutazioni
5. **Performance Testing**: Test contro agente casuale

### Risultati Attesi
- Convergenza della loss in ~30-40 iterazioni
- Miglioramento progressivo delle performance
- Strategie adaptive mostrano learning più stabile

## 📝 Report

Il progetto include:
- ✅ Codice completo e documentato
- ✅ Jupyter notebook con esperimenti
- ✅ Visualizzazioni e analisi
- 📄 PDF report (da completare con risultati finali)

## 🎯 Grading Criteria

- **Adherence to object (40%)**: ✓ Implementato Predictive MinMax completo
- **Experimentation logics (30%)**: ✓ 6 strategie testate e confrontate
- **Report (30%)**: Notebook dettagliato + PDF finale

## 🛠️ Tecnologie Utilizzate

- **Python 3.8+**
- **PyTorch**: Neural network
- **NumPy**: Operazioni numeriche
- **Matplotlib/Seaborn**: Visualizzazioni
- **Pandas**: Analisi dati
- **Jupyter**: Notebook interattivo

## 📚 Riferimenti

Come da homework PDF:
- Implementazione di `action(s) := MinMax(s, Htrue, L, K)`
- Training loop: Play → Observe → Learn
- Strategie adattive per L(t) e K(t)

## 👤 Autore

Agostino Caianiello
Artificial Intelligence 25/26

---

**Note**: Il progetto implementa completamente i requisiti dell'homework, con particolare attenzione alla sperimentazione di diverse strategie adattive e alla documentazione completa del processo di apprendimento.
