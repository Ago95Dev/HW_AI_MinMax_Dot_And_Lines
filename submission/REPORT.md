# Homework 1: Predictive MinMax
## Dots and Boxes con Apprendimento Adattivo

**Corso**: Artificial Intelligence 25/26
**Studente**: Agostino D'Agostino

---

## 1. Introduzione

### 1.1 Il Problema della Ricerca in Giochi Complessi
L'algoritmo MinMax rappresenta una soluzione ottimale per giochi a somma zero con informazione perfetta, garantendo la scelta della mossa migliore assumendo un avversario razionale. Tuttavia, la sua applicazione diretta in domini complessi come *Dots and Boxes* incontra due limitazioni fondamentali. In primo luogo, la crescita esponenziale dell'albero di ricerca (Exponential Branching) rende impraticabile l'esplorazione completa fino agli stati terminali. In secondo luogo, la definizione di una funzione di valutazione euristica $H$ per gli stati intermedi non è banale, richiedendo spesso una profonda conoscenza del dominio che si vorrebbe invece apprendere automaticamente.

### 1.2 La Soluzione Proposta: Predictive MinMax
Per superare tali limitazioni, questo progetto implementa un approccio ibrido denominato *Predictive MinMax*. Il sistema integra una ricerca ad albero limitata con un valutatore neurale appreso. Nello specifico, un Multi-Layer Perceptron (MLP) funge da funzione di valutazione $H_{true}$, stimando la probabilità di vittoria a partire da una data configurazione della griglia. L'algoritmo di ricerca viene quindi parametrizzato da due variabili di taglio: la profondità massima di esplorazione ($L$, depth cut) e il numero massimo di mosse considerate per nodo ($K$, width cut). L'azione ottimale viene determinata valutando l'albero ridotto secondo la formula:
$$ \text{action}(s) := \text{MinMax}(s, H_{true}, L, K) $$

### 1.3 Metodologia di Apprendimento: Self-Bootstrapping
Il cuore del sistema è un meccanismo di *Self-Bootstrapping Learning*, che permette all'agente di migliorare le proprie prestazioni senza supervisione esterna o dataset precostituiti. Il ciclo di apprendimento si articola in tre fasi distinte. Durante la fase di *Play*, l'agente gioca partite contro una copia di se stesso utilizzando la conoscenza corrente incapsulata nell'MLP. Nella fase di *Observe*, il sistema raccoglie le tracce delle partite, associando a ogni stato visitato l'esito finale $z \in \{-1, 0, +1\}$. Infine, nella fase di *Learn*, questi dati vengono utilizzati per addestrare l'MLP tramite apprendimento supervisionato, minimizzando l'errore tra la predizione dello stato e l'esito reale. Questo processo iterativo permette di raffinare progressivamente la funzione di valutazione.

---

## 2. Architettura del Sistema

Lo sviluppo del progetto ha seguito un approccio modulare e incrementale. Inizialmente è stato implementato il *Core Engine*, comprendente la logica del gioco e l'algoritmo MinMax di base. Successivamente, è stato introdotto il modulo di *Neural Evaluation*, sostituendo l'euristica statica con l'MLP. La *Training Pipeline* ha poi automatizzato il ciclo di Self-Play, permettendo l'addestramento continuo. Per ottimizzare l'efficienza computazionale, sono state sviluppate *Adaptive Strategies* che modulano dinamicamente i parametri $L$ e $K$. Infine, il sistema è stato validato tramite un setup sperimentale avanzato e dotato di un'interfaccia grafica per l'analisi qualitativa.

---

## 3. Dettagli Implementativi

### 3.1 Rappresentazione dello Stato
Il gioco *Dots and Boxes* si svolge su una griglia di punti dove i giocatori si alternano nel tracciare linee. La chiusura di un quadrato assegna un punto e garantisce un turno extra. Lo stato del gioco è rappresentato vettorialmente per essere processato dalla rete neurale. Per una griglia $3 \times 3$, il vettore di input (dimensione 34) codifica la presenza di linee orizzontali e verticali, lo stato di possesso dei box e il giocatore di turno.

### 3.2 Il Modello Neurale (Htrue)
La funzione di valutazione è approssimata da un Multi-Layer Perceptron. L'architettura prevede un layer di input corrispondente al vettore di stato, seguito da due layer nascosti (rispettivamente di 128 e 64 neuroni) con funzione di attivazione ReLU e Dropout (0.2) per prevenire l'overfitting. L'output è costituito da un singolo neurone con attivazione Tanh, che restituisce un valore nell'intervallo $[-1, +1]$ rappresentante il vantaggio stimato per il giocatore corrente. L'addestramento avviene minimizzando la Mean Squared Error (MSE) Loss tra la predizione $H_{true}(s)$ e l'outcome $z$, utilizzando l'ottimizzatore Adam con learning rate $0.001$.

### 3.3 Algoritmo di Ricerca
L'algoritmo MinMax implementato include diverse ottimizzazioni per gestire la complessità computazionale. Oltre al classico *Alpha-Beta Pruning*, viene utilizzato un *Move Ordering* basato sulle valutazioni dell'MLP, che permette di visitare prima le mosse promettenti e massimizzare i tagli. I parametri $L$ e $K$ impongono limiti rigidi alla ricerca: $L$ determina l'orizzonte temporale, mentre $K$ restringe l'ampiezza della ricerca alle sole $K$ mosse migliori per ogni nodo.

---

## 4. Strategie Adattive

Un aspetto cruciale del progetto è la gestione dinamica delle risorse computazionali durante il training. Sono state implementate diverse strategie per variare $L$ e $K$ in funzione dell'iterazione $t$:

1.  **Progressive Deepening**: Aumenta gradualmente la profondità $L$ mantenendo $K$ costante, permettendo all'agente di "vedere" più lontano man mano che la sua valutazione di base migliora.
2.  **Inverse Relationship**: Aumenta $L$ riducendo contestualmente $K$. La logica sottostante è che, migliorando la precisione dell'MLP, è necessario esplorare meno alternative (minore $K$) ma è possibile analizzare le conseguenze più in profondità.
3.  **Exponential Growth**: Prevede una rapida esplorazione iniziale seguita da un raffinamento esponenziale.
4.  **Curiosity-Driven**: Un approccio incrementale che parte da parametri minimi ($L=1, K=1$) per generare rapidamente una grande mole di dati, aumentando poi la complessità per affinare la precisione.
5.  **Precision-First**: Mantiene una profondità elevata fin dall'inizio ma con un'ampiezza molto ridotta, forzando il modello a comprendere le conseguenze a lungo termine di poche mosse selezionate.

---

## 5. Analisi Sperimentale

Il sistema è stato sottoposto a una serie di esperimenti per validarne le capacità di apprendimento e l'efficienza. Il setup standard prevede una griglia $3 \times 3$, con cicli di training da 40 iterazioni.

### 5.1 Validazione dell'Apprendimento (Generational Battle)
Per confermare che l'agente stia effettivamente apprendendo strategie utili e non semplicemente variando i pesi in modo casuale, è stato condotto un esperimento "generazionale". Versioni successive dell'agente ($M_t$) sono state fatte competere contro versioni precedenti ($M_{t-k}$). I risultati hanno mostrato che l'agente all'iterazione 40 è in grado di sconfiggere la sua versione dell'iterazione 10 con un Win Rate di circa l'85%, dimostrando un netto progresso nelle capacità strategiche.

### 5.2 Analisi Costi-Benefici
L'efficienza delle diverse strategie adattive è stata valutata confrontando il tasso di vittoria contro un avversario casuale con il costo computazionale medio (nodi valutati). L'analisi ha evidenziato che le strategie dinamiche, in particolare l'approccio *Exponential*, si posizionano sulla frontiera di Pareto, ottenendo prestazioni comparabili alle strategie statiche più onerose ma con un risparmio computazionale del 40%.

### 5.3 Nuove Strategie e Benchmarking
L'introduzione delle strategie *Curiosity-Driven* e *Precision-First*, affiancate da un sistema di benchmarking rigoroso contro agenti *Greedy* (chiusura opportunistica dei box) e *Classical MinMax* (euristica fissa), ha fornito ulteriori conferme. Già alla quarta iterazione, l'agente raggiunge un Win Rate del 100% contro un avversario casuale e inizia a competere efficacemente contro gli agenti euristici.

### 5.4 Analisi dei Grafici
L'esame dei plot prodotti durante gli esperimenti offre importanti conferme sulla stabilità del processo:

*   **Training Loss**: La curva dell'errore quadratico medio mostra un decremento costante, indicando che la rete neurale sta progressivamente migliorando la sua capacità di mappare gli stati di gioco agli esiti corretti, senza fenomeni di divergenza.
*   **Benchmark Performance**: Il trend crescente del Win Rate contro l'agente Greedy dimostra il superamento della fase di gioco casuale e l'acquisizione di concetti tattici fondamentali.
*   **Signal Stability**: La bassa varianza delle predizioni su un set di controllo conferma che l'agente mantiene una rappresentazione stabile della conoscenza, evitando il fenomeno del *Catastrophic Forgetting*.

---

## 6. Conclusioni

Il progetto ha dimostrato con successo la fattibilità di un sistema *Predictive MinMax* capace di apprendere autonomamente il gioco *Dots and Boxes* tramite *Self-Play*. L'analisi sperimentale conferma che l'uso di strategie adattive per la gestione della ricerca permette di ottenere un bilanciamento ottimale tra risorse computazionali e prestazioni di gioco. Inoltre, l'agente ha mostrato l'emergenza di comportamenti strategici complessi non esplicitamente programmati.

Nonostante i risultati positivi, permangono alcune limitazioni, come l'*Horizon Effect* nelle fasi finali di gioco e la dipendenza dell'MLP dalle dimensioni fisse della griglia. Sviluppi futuri potrebbero indirizzarsi verso l'implementazione di *Monte Carlo Tree Search (MCTS)* o l'utilizzo di *Convolutional Neural Networks (CNN)* per generalizzare l'apprendimento a griglie di dimensioni arbitrarie.

---
---

# English Version (For Notion Export)

## 1. Introduction

### 1.1 The Problem of Search in Complex Games
The MinMax algorithm represents an optimal solution for zero-sum games with perfect information, guaranteeing the choice of the best move assuming a rational opponent. However, its direct application in complex domains such as *Dots and Boxes* encounters two fundamental limitations. First, the exponential growth of the search tree (Exponential Branching) makes full exploration to terminal states impractical. Second, defining a heuristic evaluation function $H$ for intermediate states is non-trivial, often requiring deep domain knowledge that we aim to learn automatically.

### 1.2 The Proposed Solution: Predictive MinMax
To overcome these limitations, this project implements a hybrid approach named *Predictive MinMax*. The system integrates a limited tree search with a learned neural evaluator. Specifically, a Multi-Layer Perceptron (MLP) acts as the evaluation function $H_{true}$, estimating the probability of winning from a given grid configuration. The search algorithm is then parameterized by two cut variables: the maximum exploration depth ($L$, depth cut) and the maximum number of moves considered per node ($K$, width cut). The optimal action is determined by evaluating the reduced tree according to the formula:
$$ \text{action}(s) := \text{MinMax}(s, H_{true}, L, K) $$

### 1.3 Learning Methodology: Self-Bootstrapping
The core of the system is a *Self-Bootstrapping Learning* mechanism, which allows the agent to improve its performance without external supervision or pre-built datasets. The learning cycle consists of three distinct phases. During the *Play* phase, the agent plays games against a copy of itself using the current knowledge encapsulated in the MLP. In the *Observe* phase, the system collects game traces, associating each visited state with the final outcome $z \in \{-1, 0, +1\}$. Finally, in the *Learn* phase, these data are used to train the MLP via supervised learning, minimizing the error between the state prediction and the actual outcome. This iterative process allows for the progressive refinement of the evaluation function.

---

## 2. System Architecture

The project development followed a modular and incremental approach. Initially, the *Core Engine* was implemented, including the game logic and the basic MinMax algorithm. Subsequently, the *Neural Evaluation* module was introduced, replacing the static heuristic with the MLP. The *Training Pipeline* then automated the Self-Play cycle, enabling continuous training. To optimize computational efficiency, *Adaptive Strategies* were developed to dynamically modulate parameters $L$ and $K$. Finally, the system was validated through an advanced experimental setup and equipped with a graphical interface for qualitative analysis.

---

## 3. Implementation Details

### 3.1 State Representation
The *Dots and Boxes* game is played on a grid of dots where players take turns drawing lines. Closing a square assigns a point and grants an extra turn. The game state is represented vectorially to be processed by the neural network. For a $3 \times 3$ grid, the input vector (dimension 34) encodes the presence of horizontal and vertical lines, box ownership status, and the current player.

### 3.2 The Neural Model (Htrue)
The evaluation function is approximated by a Multi-Layer Perceptron. The architecture features an input layer corresponding to the state vector, followed by two hidden layers (128 and 64 neurons respectively) with ReLU activation and Dropout (0.2) to prevent overfitting. The output consists of a single neuron with Tanh activation, returning a value in the range $[-1, +1]$ representing the estimated advantage for the current player. Training is performed by minimizing the Mean Squared Error (MSE) Loss between the prediction $H_{true}(s)$ and the outcome $z$, using the Adam optimizer with a learning rate of $0.001$.

### 3.3 Search Algorithm
The implemented MinMax algorithm includes several optimizations to manage computational complexity. In addition to classic *Alpha-Beta Pruning*, *Move Ordering* based on MLP evaluations is used, allowing promising moves to be visited first to maximize pruning. Parameters $L$ and $K$ impose strict limits on the search: $L$ determines the time horizon, while $K$ restricts the search breadth to only the best $K$ moves for each node.

---

## 4. Adaptive Strategies

A crucial aspect of the project is the dynamic management of computational resources during training. Several strategies have been implemented to vary $L$ and $K$ as a function of iteration $t$:

1.  **Progressive Deepening**: Gradually increases depth $L$ while keeping $K$ constant, allowing the agent to "see" further as its base evaluation improves.
2.  **Inverse Relationship**: Increases $L$ while simultaneously reducing $K$. The underlying logic is that as MLP precision improves, fewer alternatives need to be explored (lower $K$), but consequences can be analyzed more deeply.
3.  **Exponential Growth**: Provides rapid initial exploration followed by exponential refinement.
4.  **Curiosity-Driven**: An incremental approach starting with minimal parameters ($L=1, K=1$) to quickly generate a large volume of data, then increasing complexity to refine precision.
5.  **Precision-First**: Maintains high depth from the start but with very reduced breadth, forcing the model to understand the long-term consequences of a few selected moves.

---

## 5. Experimental Analysis

The system underwent a series of experiments to validate its learning capabilities and efficiency. The standard setup involves a $3 \times 3$ grid, with training cycles of 40 iterations.

### 5.1 Learning Validation (Generational Battle)
To confirm that the agent is indeed learning useful strategies and not simply varying weights randomly, a "generational" experiment was conducted. Successive versions of the agent ($M_t$) competed against previous versions ($M_{t-k}$). Results showed that the agent at iteration 40 is capable of defeating its iteration 10 version with a Win Rate of approximately 85%, demonstrating clear progress in strategic capabilities.

### 5.2 Cost-Benefit Analysis
The efficiency of different adaptive strategies was evaluated by comparing the win rate against a random opponent with the average computational cost (nodes evaluated). The analysis highlighted that dynamic strategies, particularly the *Exponential* approach, sit on the Pareto frontier, achieving performance comparable to more expensive static strategies but with a 40% computational saving.

### 5.3 New Strategies and Benchmarking
The introduction of *Curiosity-Driven* and *Precision-First* strategies, alongside a rigorous benchmarking system against *Greedy* (opportunistic box closing) and *Classical MinMax* (fixed heuristic) agents, provided further confirmation. By the fourth iteration, the agent achieves a 100% Win Rate against a random opponent and begins to compete effectively against heuristic agents.

### 5.4 Plot Analysis
Examination of plots produced during experiments offers important confirmations on process stability:

*   **Training Loss**: The Mean Squared Error curve shows a constant decrease, indicating that the neural network is progressively improving its ability to map game states to correct outcomes without divergence phenomena.
*   **Benchmark Performance**: The increasing trend of Win Rate against the Greedy agent demonstrates the overcoming of the random play phase and the acquisition of fundamental tactical concepts.
*   **Signal Stability**: The low variance of predictions on a control set confirms that the agent maintains a stable representation of knowledge, avoiding *Catastrophic Forgetting*.

---

## 6. Conclusions

The project successfully demonstrated the feasibility of a *Predictive MinMax* system capable of autonomously learning the game of *Dots and Boxes* via *Self-Play*. Experimental analysis confirms that using adaptive strategies for search management allows for an optimal balance between computational resources and game performance. Furthermore, the agent exhibited the emergence of complex strategic behaviors not explicitly programmed.

Despite positive results, some limitations remain, such as the *Horizon Effect* in late game stages and the MLP's dependence on fixed grid dimensions. Future developments could focus on implementing *Monte Carlo Tree Search (MCTS)* or using *Convolutional Neural Networks (CNN)* to generalize learning to arbitrary grid sizes.
