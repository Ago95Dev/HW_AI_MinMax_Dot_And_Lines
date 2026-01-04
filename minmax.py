"""
MinMax Search Algorithm with Depth (L) and Width (K) Cuts
Implements the predictive MinMax algorithm using an MLP evaluator.
"""

from typing import Tuple, List, Optional
import numpy as np
from dots_and_boxes import DotsAndBoxes
from mlp_evaluator import MLPEvaluator
import random


class MinMaxAgent:
    """
    MinMax search agent with adaptive depth and width cuts.
    
    Uses Htrue (MLP) to evaluate leaf nodes and guide move ordering.
    """
    
    def __init__(self, evaluator: MLPEvaluator, L: int = 3, K: int = 5):
        """
        Initialize MinMax agent.
        
        Args:
            evaluator: MLP evaluator (Htrue) for position evaluation
            L: Maximum search depth (depth cut)
            K: Maximum moves to explore per node (width cut)
        """
        self.evaluator = evaluator
        self.L = L  # Depth cut
        self.K = K  # Width cut
        
        # Statistics
        self.nodes_evaluated = 0
        self.leaf_nodes = 0
    
    def select_move(self, game: DotsAndBoxes) -> Tuple[str, int, int]:
        """
        Select the best move using MinMax search.
        
        Args:
            game: Current game state
            
        Returns:
            Best move in format ('h'/'v', row, col)
        """
        self.nodes_evaluated = 0
        self.leaf_nodes = 0
        
        valid_moves = game.get_valid_moves()
        
        if len(valid_moves) == 0:
            raise ValueError("No valid moves available")
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Evaluate all moves
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        # Order moves by MLP evaluation for better pruning
        ordered_moves = self._order_moves(game, valid_moves)
        
        # Only explore top K moves
        moves_to_explore = ordered_moves[:self.K]
        
        for move in moves_to_explore:
            # Make move on a copy
            game_copy = game.copy()
            got_another = game_copy.make_move(move)
            
            # If we get another turn, we're still maximizing
            if got_another:
                value = self._minmax(game_copy, self.L - 1, alpha, beta, True)
            else:
                # Opponent's turn, so we minimize
                value = self._minmax(game_copy, self.L - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
        
        return best_move
    
    def _minmax(self, game: DotsAndBoxes, depth: int, alpha: float, 
                beta: float, maximizing: bool) -> float:
        """
        MinMax recursive search with alpha-beta pruning.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player, False if minimizing
            
        Returns:
            Evaluation score from current player's perspective
        """
        self.nodes_evaluated += 1
        
        # Base cases
        if game.is_game_over():
            self.leaf_nodes += 1
            # Return outcome from player 1's perspective
            return float(game.get_outcome(perspective=1))
        
        if depth == 0:
            # Reached depth limit, use MLP evaluation
            self.leaf_nodes += 1
            state = game.get_state_vector()
            value = self.evaluator.evaluate_state(state)
            # Adjust for current player
            if game.current_player == -1:
                value = -value
            return value
        
        valid_moves = game.get_valid_moves()
        
        if len(valid_moves) == 0:
            # No moves available (shouldn't happen)
            self.leaf_nodes += 1
            return 0.0
        
        # Order moves and apply width cut
        ordered_moves = self._order_moves(game, valid_moves)
        moves_to_explore = ordered_moves[:self.K]
        
        if maximizing:
            max_value = -float('inf')
            for move in moves_to_explore:
                game_copy = game.copy()
                got_another = game_copy.make_move(move)
                
                # If we complete a box, we get another turn (still maximizing)
                if got_another:
                    value = self._minmax(game_copy, depth - 1, alpha, beta, True)
                else:
                    value = self._minmax(game_copy, depth - 1, alpha, beta, False)
                
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_value
        else:
            min_value = float('inf')
            for move in moves_to_explore:
                game_copy = game.copy()
                got_another = game_copy.make_move(move)
                
                # If opponent completes a box, they get another turn (still minimizing)
                if got_another:
                    value = self._minmax(game_copy, depth - 1, alpha, beta, False)
                else:
                    value = self._minmax(game_copy, depth - 1, alpha, beta, True)
                
                min_value = min(min_value, value)
                beta = min(beta, value)
                
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_value
    
    def _order_moves(self, game: DotsAndBoxes, moves: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """
        Order moves by their MLP evaluation for better pruning.
        
        Args:
            game: Current game state
            moves: List of valid moves
            
        Returns:
            Ordered list of moves (best first for current player)
        """
        # Evaluate each move
        move_values = []
        
        for move in moves:
            game_copy = game.copy()
            game_copy.make_move(move)
            state = game_copy.get_state_vector()
            value = self.evaluator.evaluate_state(state)
            
            # Adjust value based on current player
            if game.current_player == -1:
                value = -value
            
            move_values.append((move, value))
        
        # Sort by value (descending - best first)
        move_values.sort(key=lambda x: x[1], reverse=True)
        
        return [move for move, _ in move_values]
    
    def update_parameters(self, L: int, K: int):
        """Update search parameters."""
        self.L = L
        self.K = K
    
    def get_statistics(self) -> dict:
        """Get search statistics."""
        return {
            'nodes_evaluated': self.nodes_evaluated,
            'leaf_nodes': self.leaf_nodes,
            'depth_limit': self.L,
            'width_limit': self.K
        }


class RandomAgent:
    """Random agent for baseline comparison."""
    
    def select_move(self, game: DotsAndBoxes) -> Tuple[str, int, int]:
        """Select a random valid move."""
        valid_moves = game.get_valid_moves()
        if len(valid_moves) == 0:
            raise ValueError("No valid moves available")
        return random.choice(valid_moves)


def test_minmax():
    """Test the MinMax agent."""
    from dots_and_boxes import DotsAndBoxes
    from mlp_evaluator import MLPEvaluator
    
    print("Testing MinMax Agent\n")
    
    # Create game and agents
    game = DotsAndBoxes(grid_size=2)
    mlp = MLPEvaluator.create_from_game(game, hidden_sizes=[64, 32])
    
    agent1 = MinMaxAgent(mlp, L=2, K=3)
    agent2 = RandomAgent()
    
    print("MinMax agent (L=2, K=3) vs Random agent\n")
    print("Initial board:")
    print(game.display())
    
    # Play a few moves
    move_count = 0
    max_moves = 5
    
    while not game.is_game_over() and move_count < max_moves:
        print(f"\n--- Move {move_count + 1} ---")
        
        if game.current_player == 1:
            move = agent1.select_move(game)
            print(f"MinMax selects: {move}")
            stats = agent1.get_statistics()
            print(f"Nodes evaluated: {stats['nodes_evaluated']}, Leaf nodes: {stats['leaf_nodes']}")
        else:
            move = agent2.select_move(game)
            print(f"Random selects: {move}")
        
        got_another = game.make_move(move)
        print(game.display())
        
        if got_another:
            print("Player gets another turn!")
        
        move_count += 1
    
    print("\n--- Test Complete ---")
    if game.is_game_over():
        winner = game.get_winner()
        if winner == 1:
            print("MinMax wins!")
        elif winner == -1:
            print("Random wins!")
        else:
            print("Tie!")
    else:
        print(f"Game stopped after {max_moves} moves for testing")


if __name__ == "__main__":
    test_minmax()
