"""
Training Loop - Self-play training for the Predictive MinMax agent
Implements the Play → Observe → Learn cycle
"""

from typing import List, Tuple
import numpy as np
from dots_and_boxes import DotsAndBoxes
from mlp_evaluator import MLPEvaluator
from minmax import MinMaxAgent
import time


class TrainingLoop:
    """
    Self-play training loop for Predictive MinMax.
    
    Iteratively plays games, collects data, and trains the MLP.
    """
    
    def __init__(self, grid_size: int = 3, hidden_sizes: List[int] = [128, 64]):
        """
        Initialize training loop.
        
        Args:
            grid_size: Size of the game grid
            hidden_sizes: MLP hidden layer sizes
        """
        self.grid_size = grid_size
        self.hidden_sizes = hidden_sizes
        
        # Create initial game and MLP
        game = DotsAndBoxes(grid_size=grid_size)
        self.mlp = MLPEvaluator.create_from_game(game, hidden_sizes=hidden_sizes)
        
        # Training statistics
        self.games_played = 0
        self.training_history = []
        self.win_rates = []  # Track performance over time
        
    def play_game(self, L: int, K: int, verbose: bool = False) -> Tuple[List[np.ndarray], int]:
        """
        Play a single self-play game.
        
        Args:
            L: Depth limit for MinMax
            K: Width limit for MinMax
            verbose: If True, print game progress
            
        Returns:
            Tuple of (states_visited, final_outcome)
        """
        game = DotsAndBoxes(grid_size=self.grid_size)
        agent = MinMaxAgent(self.mlp, L=L, K=K)
        
        states_visited = []
        
        if verbose:
            print("Starting self-play game")
            print(game.display())
        
        move_count = 0
        while not game.is_game_over():
            # Record current state
            state = game.get_state_vector()
            states_visited.append(state.copy())
            
            # Select and make move
            move = agent.select_move(game)
            got_another = game.make_move(move)
            
            move_count += 1
            
            if verbose:
                print(f"\nMove {move_count}: {move}")
                print(game.display())
                print(f"Got another turn: {got_another}")
        
        # Get final outcome from player 1's perspective
        outcome = game.get_outcome(perspective=1)
        
        if verbose:
            print(f"\nGame over! Outcome: {outcome}")
            print(f"Final score - Player 1: {game.player1_score}, Player 2: {game.player2_score}")
            print(f"Total moves: {move_count}")
            print(f"States collected: {len(states_visited)}")
        
        self.games_played += 1
        
        return states_visited, outcome
    
    def train_iteration(self, L: int, K: int, num_games: int = 10, 
                       epochs_per_batch: int = 1, verbose: bool = False) -> dict:
        """
        Run one training iteration: play multiple games and train on collected data.
        
        Args:
            L: Depth limit for MinMax
            K: Width limit for MinMax
            num_games: Number of games to play before training
            epochs_per_batch: Number of training epochs on the collected data
            verbose: If True, print detailed progress
            
        Returns:
            Dictionary with training statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Iteration {len(self.training_history) + 1}")
            print(f"Parameters: L={L}, K={K}, Games={num_games}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        # Play multiple games and collect data
        games_data = []
        outcomes_count = {1: 0, 0: 0, -1: 0}
        
        for i in range(num_games):
            if verbose and (i + 1) % max(1, num_games // 5) == 0:
                print(f"\nPlaying game {i+1}/{num_games}...")
            
            states, outcome = self.play_game(L, K, verbose=False)
            games_data.append((states, outcome))
            outcomes_count[outcome] += 1
        
        # Train on collected data
        if verbose:
            print(f"\nTraining on {len(games_data)} games...")
            print(f"Outcomes - Wins: {outcomes_count[1]}, Ties: {outcomes_count[0]}, Losses: {outcomes_count[-1]}")
        
        loss = self.mlp.train_on_games(games_data, epochs=epochs_per_batch)
        
        elapsed_time = time.time() - start_time
        
        # Record statistics
        stats = {
            'iteration': len(self.training_history) + 1,
            'L': L,
            'K': K,
            'num_games': num_games,
            'loss': loss,
            'outcomes': outcomes_count.copy(),
            'time': elapsed_time,
            'total_games': self.games_played
        }
        
        self.training_history.append(stats)
        
        if verbose:
            print(f"\nTraining Loss: {loss:.4f}")
            print(f"Average recent loss: {self.mlp.get_average_recent_loss():.4f}")
            print(f"Time: {elapsed_time:.2f}s")
        
        return stats
    
    def evaluate_performance(self, L: int, K: int, num_test_games: int = 20) -> dict:
        """
        Evaluate current agent performance.
        
        Args:
            L: Depth limit
            K: Width limit
            num_test_games: Number of games to play for evaluation
            
        Returns:
            Performance statistics
        """
        wins = 0
        ties = 0
        losses = 0
        
        for _ in range(num_test_games):
            _, outcome = self.play_game(L, K, verbose=False)
            if outcome == 1:
                wins += 1
            elif outcome == 0:
                ties += 1
            else:
                losses += 1
        
        win_rate = wins / num_test_games
        self.win_rates.append(win_rate)
        
        return {
            'wins': wins,
            'ties': ties,
            'losses': losses,
            'win_rate': win_rate,
            'num_games': num_test_games
        }
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        self.mlp.save_model(filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        self.mlp.load_model(filepath)
        print(f"Checkpoint loaded from {filepath}")
    
    def get_training_summary(self) -> dict:
        """Get summary of training progress."""
        if len(self.training_history) == 0:
            return {'message': 'No training iterations yet'}
        
        recent_loss = self.mlp.get_average_recent_loss()
        
        return {
            'total_iterations': len(self.training_history),
            'total_games': self.games_played,
            'current_loss': recent_loss,
            'training_count': self.mlp.training_count,
        }


def test_training_loop():
    """Test the training loop."""
    print("Testing Training Loop\n")
    
    # Create training loop with small grid for fast testing
    trainer = TrainingLoop(grid_size=2, hidden_sizes=[64, 32])
    
    print("Initial MLP evaluation:")
    game = DotsAndBoxes(grid_size=2)
    state = game.get_state_vector()
    initial_value = trainer.mlp.evaluate_state(state)
    print(f"Empty board value: {initial_value:.4f}\n")
    
    # Run a few training iterations
    print("Running training iterations...")
    
    # Start with small L and K
    for iteration in range(3):
        L = 2
        K = 3
        
        print(f"\n--- Iteration {iteration + 1} ---")
        stats = trainer.train_iteration(L=L, K=K, num_games=5, 
                                       epochs_per_batch=2, verbose=True)
    
    # Check value after training
    print("\n" + "="*60)
    print("After Training:")
    trained_value = trainer.mlp.evaluate_state(state)
    print(f"Empty board value: {trained_value:.4f}")
    print(f"Change from initial: {trained_value - initial_value:+.4f}")
    
    # Show training summary
    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_training_loop()
