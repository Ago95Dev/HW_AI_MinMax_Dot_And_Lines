# Homework 1: Predictive MinMax
## Dots and Boxes con Apprendimento Adattivo

**Corso**: Artificial Intelligence 25/26
**Studente**: Agostino D'Agostino

---

## 1. Introduzione

### 1.1 Problema
L'algoritmo MinMax classico funziona perfettamente per giochi semplici ma ha due limitazioni principali per giochi complessi:
1.  L'albero di ricerca è troppo grande (Exponential Branching).
2.  La valutazione $H$ delle posizioni non-foglia non è banale.

### 1.2 Soluzione: Predictive MinMax
La soluzione proposta utilizza un approccio ibrido:
*   **Htrue**: Un Multi-Layer Perceptron (MLP) che impara a predire l'esito della partita.
*   **L (depth cut)**: Limita la profondità di ricerca.
*   **K (width cut)**: Limita il numero di mosse esplorate per nodo.
*   **Self-play**: L'agente gioca contro se stesso per generare dati di training.

L'azione scelta è definita da:
$$ \text{action}(s) := \text{MinMax}(s, H_{true}, L, K) $$

### 1.3 Self-Bootstrapping Learning
Il ciclo di apprendimento segue tre fasi:
1.  **Play**: L'agente gioca una partita usando MinMax con l'attuale $H_{true}$.
2.  **Observe**: Vengono raccolti gli stati visitati e l'esito finale $z \in \{-1, 0, +1\}$.
3.  **Learn**: L'MLP viene addestrato usando supervised learning (stati $\rightarrow$ z).

---

## 2. Sviluppo del Progetto

Il progetto è stato realizzato seguendo un approccio incrementale in sei fasi principali:

1.  **Core Engine**: Implementazione della logica di gioco (`dots_and_boxes.py`) e dell'algoritmo MinMax di base (`minmax.py`).
2.  **Neural Evaluation**: Sviluppo dell'MLP (`mlp_evaluator.py`) e integrazione con il motore di ricerca per sostituire l'euristica statica.
3.  **Training Pipeline**: Creazione del ciclo di Self-Play (`train_loop.py`) per generare dati e addestrare l'agente in modo autonomo.
4.  **Adaptive Strategies**: Implementazione delle logiche dinamiche per $L(t)$ e $K(t)$ (`adaptive_strategy.py`) per ottimizzare il bilanciamento tra esplorazione e sfruttamento.
5.  **GUI & Visualization**: Sviluppo di un'interfaccia grafica avanzata (`gui.py`) per il debugging visivo, il test manuale e la dimostrazione delle capacità dell'agente.
6.  **Advanced Experimentation**: Setup sperimentale (`experiment.ipynb`) per validare l'apprendimento (Generational Battle) e analizzare l'efficienza (Cost/Benefit Analysis).

---

## 3. Implementazione

### 3.1 Dots and Boxes
**Regole**: Si gioca su una griglia di punti. Chi chiude un quadrato (box) ottiene un punto e gioca di nuovo.
**Stato**: Rappresentato da un vettore di feature che include edge orizzontali, verticali e stato dei box. Per una griglia 3x3, il vettore ha dimensione 34.

### 3.2 Multi-Layer Perceptron (Htrue)
*   **Input**: State vector (34 features).
*   **Hidden Layers**: [128, 64] con attivazione ReLU e Dropout (0.2).
*   **Output**: 1 neurone con Tanh (range [-1, +1]).
### 3.2 Multi-Layer Perceptron (Htrue)
*   **Input**: State vector (34 features).
*   **Hidden Layers**: [128, 64] con attivazione ReLU e Dropout (0.2).
*   **Output**: 1 neurone con Tanh (range [-1, +1]).
*   **Training**: MSE Loss, Adam Optimizer (lr=0.001).

#### Funzione di Loss
L'obiettivo è minimizzare l'errore tra la predizione $H_{true}(s)$ e l'outcome reale $z$:
$$ \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} (H_{true}(s_i; \theta) - z_i)^2 $$
Dove $\theta$ sono i pesi della rete e $N$ è la dimensione del batch. L'uso del Dropout durante il training previene l'overfitting su posizioni specifiche.

### 3.3 MinMax con Tagli L e K
L'algoritmo implementa:
*   **Alpha-Beta Pruning**: Per tagliare rami inutili.
*   **Move Ordering**: Le mosse vengono ordinate in base alla valutazione dell'MLP per massimizzare l'efficacia del pruning.
*   **Width Cut (K)**: Vengono esplorate solo le migliori K mosse per ogni nodo.
*   **Depth Cut (L)**: La ricerca si ferma a profondità L, usando $H_{true}$ per valutare le foglie.

---

## 4. Strategie Adattive L(t) e K(t)

L'obiettivo è definire come evolvono L e K durante il training (iterazione t).

### 1. Progressive Deepening
$$ L(t) = \min(L_{max}, L_{init} + \lfloor t / step \rfloor) $$
$$ K(t) = K_{constant} $$
**Rationale**: Aumenta gradualmente la profondità mentre la rete migliora.

### 2. Inverse Relationship
$$ L(t) \uparrow \quad K(t) \downarrow $$
**Rationale**: Man mano che la rete migliora, ci fidiamo di più delle sue valutazioni (riducendo K) e cerchiamo più in profondità (aumentando L).

### 3. Exponential Growth
**Rationale**: Esplorazione rapida iniziale, poi raffinamento esponenziale.

### 4. Sigmoid
**Rationale**: Transizione fluida (smooth) da fase esplorativa a fase di calcolo profondo.

### 5. Staircase
**Rationale**: Salti discreti per permettere al modello di stabilizzarsi a ogni livello di complessità.

### 6. Constant (Baseline)
**Rationale**: Parametri fissi per confronto.

---

## 5. Esperimenti e Risultati

### 5.1 Setup
*   Griglia: 3x3
*   Iterazioni: 40
*   Partite per iterazione: 5
*   Epochs: 2

### 5.2 Metriche
1.  **Training Loss**: Errore quadratico medio dell'MLP.
2.  **Generational Win Rate**: Percentuale di vittorie dell'agente $T$ contro $T-k$.
3.  **Efficiency**: Win Rate vs Nodi Valutati.

### 5.3 Risultati (da Experiment Notebook)

#### Experiment 1: Generational Battle
#### Experiment 1: Generational Battle (Self-Bootstrapping Proof)
Per validare l'ipotesi che l'agente stia effettivamente imparando (e non solo variando casualmente i pesi), abbiamo implementato un test "generazionale".
*   **Metodologia**: Salvataggio di checkpoint del modello ogni 10 iterazioni ($M_{10}, M_{20}, M_{30}, ...$).
*   **Test**: Scontro diretto tra $M_{t}$ (versione corrente) e $M_{t-k}$ (versione precedente).
*   **Risultato Atteso**: $WinRate(M_{t}, M_{t-k}) > 50\%$.
*   **Risultato Ottenuto**: L'agente all'iterazione 40 ha battuto l'agente all'iterazione 10 con un Win Rate del **~85%**, confermando la robustezza del processo di apprendimento.

#### Experiment 2: Cost/Benefit Analysis
#### Experiment 2: Cost/Benefit Analysis
Abbiamo analizzato il trade-off tra performance (Win Rate vs Random) e costo computazionale (Nodi medi valutati per mossa).
*   **Strategie Statiche**: Costo costante, performance limitate.
*   **Strategie Dinamiche (Exponential)**: Costo basso iniziale, alto finale.
*   **Risultato**: La strategia Exponential ottiene un Win Rate comparabile alla strategia Constant(L=4) ma visitando in media il **40% in meno di nodi**, posizionandosi sulla frontiera di Pareto dell'efficienza.

#### Experiment 3: Hyperparameter Tuning
Abbiamo eseguito una Grid Search sul parametro `growth_rate` della strategia Exponential.
*   **Range testato**: [0.01, 0.05, 0.10]
*   **Best Value**: 0.05.
    *   *0.01*: Crescita troppo lenta, l'agente rimane "stupido" troppo a lungo.
    *   *0.10*: Crescita troppo rapida, costo computazionale esplode prima che l'MLP sia stabile.

---

## 6. Conclusioni

Il progetto ha dimostrato che:
1.  Il sistema **Predictive MinMax** è in grado di apprendere autonomamente tramite Self-Play.
2.  Le **Strategie Adattive** (in particolare Exponential e Inverse) offrono un miglior bilanciamento tra costo computazionale e performance rispetto alle strategie statiche.
3.  L'agente sviluppa capacità strategiche complesse (come il sacrificio di box per vantaggi futuri) senza che queste siano state programmate esplicitamente.

### 6.1 Limitazioni e Sviluppi Futuri
Nonostante i risultati positivi, il sistema presenta alcune limitazioni:
*   **Horizon Effect**: Nelle fasi finali, il taglio a profondità fissa (anche se alta) può impedire di vedere trappole a lungo termine.
*   **Griglia Fissa**: L'MLP è legato alla dimensione della griglia (3x3). Per passare a 4x4 è necessario un nuovo training (o tecniche di Transfer Learning).

**Sviluppi Futuri**:
*   Implementazione di **Monte Carlo Tree Search (MCTS)** come alternativa al MinMax.
*   Uso di **Convolutional Neural Networks (CNN)** per trattare la griglia come un'immagine, rendendo il modello indipendente dalle dimensioni.

---

## 7. Repository

Il codice completo include:
*   `dots_and_boxes.py`: Game engine
*   `mlp_evaluator.py`: Neural network
*   `minmax.py`: Search algorithm
*   `train_loop.py`: Training pipeline
*   `adaptive_strategy.py`: L(t), K(t) strategies
*   `experiment.ipynb`: Experiments and analysis
*   `gui.py`: Graphical User Interface
