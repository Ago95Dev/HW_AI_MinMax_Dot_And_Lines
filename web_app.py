"""
Flask Web Application for Dots and Boxes Game
Provides REST API for game management, AI opponents, and training visualization
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time

# Import game components
from dots_and_boxes import DotsAndBoxes
from mlp_evaluator import MLPEvaluator
from minmax import MinMaxAgent
from train_loop import TrainingLoop
from adaptive_strategy import get_all_strategies

app = Flask(__name__)
CORS(app)

# Game sessions storage
game_sessions = {}
training_sessions = {}


class GameSession:
    """Manages a single game session"""
    
    def __init__(self, session_id, grid_size=3, ai_strategy='constant', ai_L=3, ai_K=5):
        self.session_id = session_id
        self.game = DotsAndBoxes(grid_size=grid_size)
        self.grid_size = grid_size
        self.ai_strategy = ai_strategy
        self.ai_L = ai_L
        self.ai_K = ai_K
        self.mode = 'human_vs_ai'  # 'human_vs_ai', 'ai_vs_ai', 'human_vs_human'
        self.history = []
        self.created_at = datetime.now()
        
        # Initialize AI
        self.mlp = MLPEvaluator.create_from_game(self.game, hidden_sizes=[64, 32])
        self.ai_agent = MinMaxAgent(self.mlp, L=ai_L, K=ai_K)
    
    def make_move(self, move):
        """Make a move and return the result"""
        if move not in self.game.get_valid_moves():
            return {'success': False, 'error': 'Invalid move'}
        
        old_score = (self.game.player1_score, self.game.player2_score)
        self.game.make_move(move)
        new_score = (self.game.player1_score, self.game.player2_score)
        
        got_box = new_score != old_score
        
        self.history.append({
            'move': move,
            'player': 3 - self.game.current_player if not got_box else self.game.current_player,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'success': True,
            'state': self.get_state(),
            'got_box': got_box
        }
    
    def get_ai_move(self):
        """Get AI's move"""
        move = self.ai_agent.select_move(self.game)
        return move
    
    def get_state(self):
        """Get current game state"""
        return {
            'grid_size': self.grid_size,
            'h_edges': self.game.horizontal_edges.astype(int).tolist(),
            'v_edges': self.game.vertical_edges.astype(int).tolist(),
            'boxes': self.game.boxes.tolist(),
            'scores': [self.game.player1_score, self.game.player2_score],
            'current_player': int(self.game.current_player),
            'valid_moves': [list(move) for move in self.game.get_valid_moves()],
            'is_over': bool(self.game.is_game_over()),
            'winner': int(self.game.get_winner()) if self.game.is_game_over() else None
        }


@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Create a new game session"""
    data = request.json
    grid_size = data.get('grid_size', 3)
    ai_strategy = data.get('ai_strategy', 'constant')
    ai_L = data.get('ai_L', 3)
    ai_K = data.get('ai_K', 5)
    mode = data.get('mode', 'human_vs_ai')
    
    session_id = f"game_{len(game_sessions)}_{int(time.time() * 1000)}"
    session = GameSession(session_id, grid_size, ai_strategy, ai_L, ai_K)
    session.mode = mode
    
    game_sessions[session_id] = session
    
    return jsonify({
        'session_id': session_id,
        'state': session.get_state()
    })


@app.route('/api/game/<session_id>/state', methods=['GET'])
def get_game_state(session_id):
    """Get current game state"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    return jsonify(session.get_state())


@app.route('/api/game/<session_id>/move', methods=['POST'])
def make_move(session_id):
    """Make a move"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    data = request.json
    move = tuple(data['move']) if isinstance(data['move'], list) else data['move']
    
    result = session.make_move(move)
    return jsonify(result)


@app.route('/api/game/<session_id>/ai_move', methods=['POST'])
def ai_move(session_id):
    """Get and execute AI move"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    
    if session.game.is_game_over():
        return jsonify({'error': 'Game is over'}), 400
    
    # Get AI move
    ai_move = session.get_ai_move()
    
    # Execute move
    result = session.make_move(ai_move)
    result['ai_move'] = ai_move
    
    return jsonify(result)


@app.route('/api/game/<session_id>/reset', methods=['POST'])
def reset_game(session_id):
    """Reset game to initial state"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    session.game = DotsAndBoxes(grid_size=session.grid_size)
    session.history = []
    session.mlp = MLPEvaluator.create_from_game(session.game, hidden_sizes=[64, 32])
    session.ai_agent = MinMaxAgent(session.mlp, L=session.ai_L, K=session.ai_K)
    
    return jsonify(session.get_state())


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get available AI strategies"""
    strategies = get_all_strategies()
    strategy_info = {}
    
    for name, strategy in strategies.items():
        L, K = strategy.get_params(0)
        strategy_info[name] = {
            'name': name,
            'initial_L': L,
            'initial_K': K,
            'description': strategy.__class__.__name__
        }
    
    return jsonify(strategy_info)


@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start a training session"""
    data = request.json
    grid_size = data.get('grid_size', 2)
    strategy_name = data.get('strategy', 'constant')
    num_iterations = data.get('num_iterations', 20)
    
    session_id = f"train_{int(time.time() * 1000)}"
    
    # Create training loop
    trainer = TrainingLoop(grid_size=grid_size, hidden_sizes=[64, 32])
    
    training_sessions[session_id] = {
        'trainer': trainer,
        'strategy': strategy_name,
        'num_iterations': num_iterations,
        'progress': [],
        'status': 'running',
        'current_iteration': 0
    }
    
    # Start training in background thread
    def run_training():
        strategies = get_all_strategies()
        strategy = strategies[strategy_name]
        
        for i in range(num_iterations):
            L, K = strategy.get_params(i)
            
            # Play games
            states, outcome = trainer.play_game(L=L, K=K, verbose=False)
            
            # Train
            loss = trainer.train_batch(states, outcome, epochs=2)
            
            training_sessions[session_id]['progress'].append({
                'iteration': i,
                'loss': float(loss),
                'L': L,
                'K': K
            })
            training_sessions[session_id]['current_iteration'] = i + 1
        
        training_sessions[session_id]['status'] = 'completed'
    
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'session_id': session_id,
        'status': 'started'
    })


@app.route('/api/train/<session_id>/progress', methods=['GET'])
def get_training_progress(session_id):
    """Get training progress"""
    if session_id not in training_sessions:
        return jsonify({'error': 'Training session not found'}), 404
    
    session = training_sessions[session_id]
    return jsonify({
        'status': session['status'],
        'current_iteration': session['current_iteration'],
        'total_iterations': session['num_iterations'],
        'progress': session['progress']
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Dots and Boxes Web Server")
    print("=" * 60)
    print("\nServer starting on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  - GET  /                         Main interface")
    print("  - POST /api/new_game             Create new game")
    print("  - GET  /api/game/<id>/state      Get game state")
    print("  - POST /api/game/<id>/move       Make move")
    print("  - POST /api/game/<id>/ai_move    Get AI move")
    print("  - GET  /api/strategies           List strategies")
    print("  - POST /api/train/start          Start training")
    print("\n" + "=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
