// Dots and Boxes Game - Frontend Logic

class DotsAndBoxesGame {
    constructor() {
        this.sessionId = null;
        this.gameState = null;
        this.mode = 'human_vs_ai';
        this.gridSize = 3;
        this.aiStrategy = 'constant';
        this.aiL = 3;
        this.aiK = 5;
        this.isAITurn = false;

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Mode selector
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.mode = e.target.dataset.mode;
            });
        });

        // Settings
        document.getElementById('grid-size').addEventListener('change', (e) => {
            this.gridSize = parseInt(e.target.value);
        });

        document.getElementById('ai-strategy').addEventListener('change', (e) => {
            this.aiStrategy = e.target.value;
        });

        document.getElementById('ai-L').addEventListener('input', (e) => {
            this.aiL = parseInt(e.target.value);
            document.getElementById('ai-L-value').textContent = this.aiL;
        });

        document.getElementById('ai-K').addEventListener('input', (e) => {
            this.aiK = parseInt(e.target.value);
            document.getElementById('ai-K-value').textContent = this.aiK;
        });

        // Buttons
        document.getElementById('new-game-btn').addEventListener('click', () => this.newGame());
        document.getElementById('play-again-btn').addEventListener('click', () => this.newGame());
        document.getElementById('train-btn').addEventListener('click', () => this.startTraining());
    }

    async newGame() {
        try {
            const response = await fetch('/api/new_game', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    grid_size: this.gridSize,
                    ai_strategy: this.aiStrategy,
                    ai_L: this.aiL,
                    ai_K: this.aiK,
                    mode: this.mode
                })
            });

            const data = await response.json();
            this.sessionId = data.session_id;
            this.gameState = data.state;

            this.renderBoard();
            this.updateUI();

            // Hide game over overlay
            document.getElementById('game-over-overlay').classList.add('hidden');

            // If AI vs AI mode, start the game automatically
            if (this.mode === 'ai_vs_ai') {
                this.playAIvsAI();
            }
        } catch (error) {
            console.error('Error creating new game:', error);
            alert('Errore nella creazione della partita');
        }
    }

    renderBoard() {
        const board = document.getElementById('game-board');
        board.innerHTML = '';

        const n = this.gameState.grid_size;
        const cellSize = Math.min(80, 500 / (n + 1));

        // Create grid layout
        board.style.gridTemplateColumns = `repeat(${2 * n + 1}, ${cellSize}px)`;
        board.style.gridTemplateRows = `repeat(${2 * n + 1}, ${cellSize}px)`;

        // Generate board elements
        for (let row = 0; row <= 2 * n; row++) {
            for (let col = 0; col <= 2 * n; col++) {
                const element = document.createElement('div');

                if (row % 2 === 0 && col % 2 === 0) {
                    // Dot
                    element.className = 'dot';
                } else if (row % 2 === 0 && col % 2 === 1) {
                    // Horizontal edge
                    const edgeRow = row / 2;
                    const edgeCol = (col - 1) / 2;
                    element.className = 'edge horizontal';
                    element.innerHTML = '<div class="edge-line"></div>';
                    element.dataset.type = 'h';
                    element.dataset.row = edgeRow;
                    element.dataset.col = edgeCol;

                    // Check if filled
                    if (this.gameState.h_edges[edgeRow][edgeCol] === 1) {
                        element.classList.add('filled');
                    } else {
                        element.addEventListener('click', () => this.makeMove('h', edgeRow, edgeCol));
                    }
                } else if (row % 2 === 1 && col % 2 === 0) {
                    // Vertical edge
                    const edgeRow = (row - 1) / 2;
                    const edgeCol = col / 2;
                    element.className = 'edge vertical';
                    element.innerHTML = '<div class="edge-line"></div>';
                    element.dataset.type = 'v';
                    element.dataset.row = edgeRow;
                    element.dataset.col = edgeCol;

                    // Check if filled
                    if (this.gameState.v_edges[edgeRow][edgeCol] === 1) {
                        element.classList.add('filled');
                    } else {
                        element.addEventListener('click', () => this.makeMove('v', edgeRow, edgeCol));
                    }
                } else {
                    // Box
                    const boxRow = (row - 1) / 2;
                    const boxCol = (col - 1) / 2;
                    element.className = 'box';

                    const boxOwner = this.gameState.boxes[boxRow][boxCol];
                    if (boxOwner === 1) {
                        element.classList.add('player1');
                        element.textContent = '1';
                    } else if (boxOwner === 2) {
                        element.classList.add('player2');
                        element.textContent = '2';
                    }
                }

                board.appendChild(element);
            }
        }
    }

    async makeMove(type, row, col) {
        if (this.isAITurn) return;
        if (this.gameState.is_over) return;

        try {
            const response = await fetch(`/api/game/${this.sessionId}/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ move: [type, row, col] })
            });

            const data = await response.json();

            if (data.success) {
                this.gameState = data.state;
                this.renderBoard();
                this.updateUI();

                if (this.gameState.is_over) {
                    this.showGameOver();
                } else if (this.mode === 'human_vs_ai' && !data.got_box) {
                    // AI's turn
                    setTimeout(() => this.makeAIMove(), 500);
                }
            } else {
                console.error('Invalid move:', data.error);
            }
        } catch (error) {
            console.error('Error making move:', error);
        }
    }

    async makeAIMove() {
        this.isAITurn = true;
        document.getElementById('ai-thinking').classList.remove('hidden');

        try {
            const response = await fetch(`/api/game/${this.sessionId}/ai_move`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.gameState = data.state;
                this.renderBoard();
                this.updateUI();

                if (this.gameState.is_over) {
                    this.showGameOver();
                } else if (data.got_box) {
                    // AI got a box, play again
                    setTimeout(() => this.makeAIMove(), 500);
                }
            }
        } catch (error) {
            console.error('Error making AI move:', error);
        } finally {
            this.isAITurn = false;
            document.getElementById('ai-thinking').classList.add('hidden');
        }
    }

    async playAIvsAI() {
        if (this.gameState.is_over) {
            this.showGameOver();
            return;
        }

        await this.makeAIMove();

        if (!this.gameState.is_over) {
            setTimeout(() => this.playAIvsAI(), 800);
        }
    }

    updateUI() {
        // Update scores
        document.getElementById('score-1').textContent = this.gameState.scores[0];
        document.getElementById('score-2').textContent = this.gameState.scores[1];

        // Update current player
        const playerName = this.gameState.current_player === 1 ? 'Giocatore 1' :
            this.mode === 'human_vs_ai' ? 'AI' : 'Giocatore 2';
        document.getElementById('current-player-name').textContent = playerName;

        // Update moves count
        document.getElementById('moves-count').textContent = this.gameState.valid_moves.length;
    }

    showGameOver() {
        const overlay = document.getElementById('game-over-overlay');
        const winnerText = document.getElementById('winner-text');

        const score1 = this.gameState.scores[0];
        const score2 = this.gameState.scores[1];

        if (score1 > score2) {
            winnerText.textContent = 'Giocatore 1 Vince!';
            winnerText.style.color = 'var(--player1-color)';
        } else if (score2 > score1) {
            const player2Name = this.mode === 'human_vs_ai' ? 'AI' : 'Giocatore 2';
            winnerText.textContent = `${player2Name} Vince!`;
            winnerText.style.color = 'var(--player2-color)';
        } else {
            winnerText.textContent = 'Pareggio!';
            winnerText.style.color = 'var(--accent)';
        }

        document.getElementById('final-score-1').textContent = score1;
        document.getElementById('final-score-2').textContent = score2;

        overlay.classList.remove('hidden');
    }

    async startTraining() {
        const iterations = parseInt(document.getElementById('train-iterations').value);
        const progressEl = document.getElementById('training-progress');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const trainBtn = document.getElementById('train-btn');

        trainBtn.disabled = true;
        trainBtn.textContent = '⏳ Training in corso...';
        progressEl.classList.remove('hidden');

        try {
            // Start training
            const response = await fetch('/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    grid_size: this.gridSize,
                    strategy: this.aiStrategy,
                    num_iterations: iterations
                })
            });

            const data = await response.json();
            const trainingId = data.session_id;

            // Poll for progress
            const pollInterval = setInterval(async () => {
                const progressResponse = await fetch(`/api/train/${trainingId}/progress`);
                const progressData = await progressResponse.json();

                const percent = (progressData.current_iteration / progressData.total_iterations) * 100;
                progressFill.style.width = `${percent}%`;
                progressText.textContent = `${progressData.current_iteration}/${progressData.total_iterations}`;

                if (progressData.status === 'completed') {
                    clearInterval(pollInterval);
                    trainBtn.disabled = false;
                    trainBtn.textContent = '🧠 Avvia Training';

                    setTimeout(() => {
                        progressEl.classList.add('hidden');
                        alert('Training completato!');
                    }, 1000);
                }
            }, 500);

        } catch (error) {
            console.error('Error starting training:', error);
            trainBtn.disabled = false;
            trainBtn.textContent = '🧠 Avvia Training';
            progressEl.classList.add('hidden');
            alert('Errore durante il training');
        }
    }
}

// Initialize game when page loads
let game;
window.addEventListener('DOMContentLoaded', () => {
    game = new DotsAndBoxesGame();
    game.newGame();
});
