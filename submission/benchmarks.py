"""
Benchmark Agents
Standard opponents for evaluating the Predictive MinMax agent.
"""

from typing import Tuple, List
import random
from dots_and_boxes import DotsAndBoxes
from minmax import RandomAgent

class GreedyAgent:
    """
    Greedy Agent:
    - If a box can be closed, close it.
    - Otherwise, play randomly.
    """
    
    def select_move(self, game: DotsAndBoxes) -> Tuple[str, int, int]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Check for box closing moves
        for move in valid_moves:
            # Check if this move completes a box
            # We can use the game's internal check without modifying state
            # But check_box_completion needs the move to be made? 
            # No, check_box_completion checks if the move *would* complete a box
            # Wait, check_box_completion in DotsAndBoxes checks if a box IS complete
            # It assumes the move has just been made or we are checking the board state.
            
            # Let's simulate the move
            game_copy = game.copy()
            got_another = game_copy.make_move(move)
            if got_another:
                return move
        
        # If no box closing move, play random
        return random.choice(valid_moves)


class ClassicalMinMaxAgent:
    """
    Classical MinMax Agent:
    - Uses standard MinMax with a fixed heuristic.
    - Heuristic: (My Boxes - Opponent Boxes).
    - No Neural Network.
    """
    
    def __init__(self, depth: int = 2):
        self.depth = depth
        self.nodes_evaluated = 0
    
    def select_move(self, game: DotsAndBoxes) -> Tuple[str, int, int]:
        self.nodes_evaluated = 0
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves")
        
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in valid_moves:
            game_copy = game.copy()
            got_another = game_copy.make_move(move)
            
            if got_another:
                value = self._minmax(game_copy, self.depth, alpha, beta, True)
            else:
                value = self._minmax(game_copy, self.depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
        
        return best_move
    
    def _minmax(self, game: DotsAndBoxes, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        self.nodes_evaluated += 1
        
        if game.is_game_over() or depth <= 0:
            # Heuristic: Score difference from perspective of current player
            # If maximizing (original player), we want (P1 - P2)
            # If minimizing (opponent), we want (P1 - P2) to be small
            
            # Actually, let's just return P1 - P2 always, and maximizing handles it
            score_diff = game.player1_score - game.player2_score
            
            # If game over, weight it heavily
            if game.is_game_over():
                if score_diff > 0: return 100 + score_diff
                if score_diff < 0: return -100 + score_diff
                return 0
            
            return score_diff
        
        valid_moves = game.get_valid_moves()
        
        if maximizing:
            max_value = -float('inf')
            for move in valid_moves:
                game_copy = game.copy()
                got_another = game_copy.make_move(move)
                
                if got_another:
                    value = self._minmax(game_copy, depth, alpha, beta, True)
                else:
                    value = self._minmax(game_copy, depth - 1, alpha, beta, False)
                
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                if beta <= alpha: break
            return max_value
        else:
            min_value = float('inf')
            for move in valid_moves:
                game_copy = game.copy()
                got_another = game_copy.make_move(move)
                
                if got_another:
                    value = self._minmax(game_copy, depth, alpha, beta, False)
                else:
                    value = self._minmax(game_copy, depth - 1, alpha, beta, True)
                
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha: break
            return min_value

def test_benchmarks():
    """Test benchmark agents."""
    print("Testing Benchmark Agents\n")
    
    game = DotsAndBoxes(grid_size=2)
    greedy = GreedyAgent()
    classical = ClassicalMinMaxAgent(depth=2)
    random_agent = RandomAgent()
    
    print("Greedy vs Random")
    moves = 0
    while not game.is_game_over() and moves < 20:
        if game.current_player == 1:
            move = greedy.select_move(game)
        else:
            move = random_agent.select_move(game)
        game.make_move(move)
        moves += 1
    print(f"Game over: {game.is_game_over()}, Winner: {game.get_winner()}")
    
    print("\nClassical vs Random")
    game = DotsAndBoxes(grid_size=2)
    moves = 0
    while not game.is_game_over() and moves < 20:
        if game.current_player == 1:
            move = classical.select_move(game)
        else:
            move = random_agent.select_move(game)
        game.make_move(move)
        moves += 1
    print(f"Game over: {game.is_game_over()}, Winner: {game.get_winner()}")

if __name__ == "__main__":
    test_benchmarks()
