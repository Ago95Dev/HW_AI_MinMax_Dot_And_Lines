"""
Dots and Boxes Game Implementation
A complete implementation of the Dots and Boxes game with state management,
move validation, and game logic.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from copy import deepcopy


class DotsAndBoxes:
    """
    Dots and Boxes game implementation.
    
    Grid structure:
    - For an n×n game, we have (n+1)×(n+1) dots
    - Horizontal edges: n×(n+1) edges
    - Vertical edges: (n+1)×n edges
    - Boxes: n×n boxes
    """
    
    def __init__(self, grid_size: int = 3):
        """
        Initialize the game.
        
        Args:
            grid_size: Size of the grid (number of boxes per side)
        """
        self.grid_size = grid_size
        self.dots = grid_size + 1
        
        # Edges are represented as boolean matrices
        # horizontal_edges[i][j] represents edge between dot(i,j) and dot(i,j+1)
        self.horizontal_edges = np.zeros((self.dots, self.grid_size), dtype=bool)
        
        # vertical_edges[i][j] represents edge between dot(i,j) and dot(i+1,j)
        self.vertical_edges = np.zeros((self.grid_size, self.dots), dtype=bool)
        
        # Box ownership: 0 = unclaimed, 1 = player 1, -1 = player 2
        self.boxes = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Current player: 1 or -1
        self.current_player = 1
        
        # Game statistics
        self.moves_count = 0
        self.player1_score = 0
        self.player2_score = 0
    
    def copy(self) -> 'DotsAndBoxes':
        """Create a deep copy of the current game state."""
        new_game = DotsAndBoxes(self.grid_size)
        new_game.horizontal_edges = self.horizontal_edges.copy()
        new_game.vertical_edges = self.vertical_edges.copy()
        new_game.boxes = self.boxes.copy()
        new_game.current_player = self.current_player
        new_game.moves_count = self.moves_count
        new_game.player1_score = self.player1_score
        new_game.player2_score = self.player2_score
        return new_game
    
    def get_valid_moves(self) -> List[Tuple[str, int, int]]:
        """
        Get all valid moves in the current state.
        
        Returns:
            List of moves in format ('h', row, col) or ('v', row, col)
            'h' = horizontal edge, 'v' = vertical edge
        """
        moves = []
        
        # Check horizontal edges
        for i in range(self.dots):
            for j in range(self.grid_size):
                if not self.horizontal_edges[i][j]:
                    moves.append(('h', i, j))
        
        # Check vertical edges
        for i in range(self.grid_size):
            for j in range(self.dots):
                if not self.vertical_edges[i][j]:
                    moves.append(('v', i, j))
        
        return moves
    
    def is_valid_move(self, move: Tuple[str, int, int]) -> bool:
        """Check if a move is valid."""
        edge_type, row, col = move
        
        if edge_type == 'h':
            if row < 0 or row >= self.dots or col < 0 or col >= self.grid_size:
                return False
            return not self.horizontal_edges[row][col]
        elif edge_type == 'v':
            if row < 0 or row >= self.grid_size or col < 0 or col >= self.dots:
                return False
            return not self.vertical_edges[row][col]
        
        return False
    
    def check_box_completion(self, move: Tuple[str, int, int]) -> List[Tuple[int, int]]:
        """
        Check which boxes (if any) are completed by a move.
        
        Returns:
            List of (row, col) tuples of completed boxes
        """
        edge_type, row, col = move
        completed_boxes = []
        
        if edge_type == 'h':
            # Horizontal edge can complete boxes above and below
            # Box above: (row-1, col)
            if row > 0:
                if self._is_box_complete(row - 1, col):
                    completed_boxes.append((row - 1, col))
            
            # Box below: (row, col)
            if row < self.grid_size:
                if self._is_box_complete(row, col):
                    completed_boxes.append((row, col))
        
        else:  # edge_type == 'v'
            # Vertical edge can complete boxes left and right
            # Box left: (row, col-1)
            if col > 0:
                if self._is_box_complete(row, col - 1):
                    completed_boxes.append((row, col - 1))
            
            # Box right: (row, col)
            if col < self.grid_size:
                if self._is_box_complete(row, col):
                    completed_boxes.append((row, col))
        
        return completed_boxes
    
    def _is_box_complete(self, box_row: int, box_col: int) -> bool:
        """
        Check if a specific box has all 4 edges.
        
        Box at (box_row, box_col) is bounded by:
        - Top edge: horizontal_edges[box_row][box_col]
        - Bottom edge: horizontal_edges[box_row+1][box_col]
        - Left edge: vertical_edges[box_row][box_col]
        - Right edge: vertical_edges[box_row][box_col+1]
        """
        if box_row < 0 or box_row >= self.grid_size or box_col < 0 or box_col >= self.grid_size:
            return False
        
        # Check if box is already claimed
        if self.boxes[box_row][box_col] != 0:
            return False
        
        top = self.horizontal_edges[box_row][box_col]
        bottom = self.horizontal_edges[box_row + 1][box_col]
        left = self.vertical_edges[box_row][box_col]
        right = self.vertical_edges[box_row][box_col + 1]
        
        return top and bottom and left and right
    
    def make_move(self, move: Tuple[str, int, int]) -> bool:
        """
        Make a move and update the game state.
        
        Returns:
            True if the player gets another turn (completed a box), False otherwise
        """
        if not self.is_valid_move(move):
            raise ValueError(f"Invalid move: {move}")
        
        edge_type, row, col = move
        
        # Place the edge
        if edge_type == 'h':
            self.horizontal_edges[row][col] = True
        else:
            self.vertical_edges[row][col] = True
        
        self.moves_count += 1
        
        # Check for completed boxes
        completed_boxes = self.check_box_completion(move)
        
        # Claim completed boxes for current player
        for box_row, box_col in completed_boxes:
            self.boxes[box_row][box_col] = self.current_player
            if self.current_player == 1:
                self.player1_score += 1
            else:
                self.player2_score += 1
        
        # If boxes were completed, player gets another turn
        got_another_turn = len(completed_boxes) > 0
        
        # Switch player if no boxes were completed
        if not got_another_turn:
            self.current_player = -self.current_player
        
        return got_another_turn
    
    def is_game_over(self) -> bool:
        """Check if the game is over (all boxes claimed)."""
        return np.all(self.boxes != 0)
    
    def get_winner(self) -> int:
        """
        Get the winner of the game.
        
        Returns:
            1 if player 1 wins, -1 if player 2 wins, 0 if tie
        """
        if not self.is_game_over():
            return 0
        
        if self.player1_score > self.player2_score:
            return 1
        elif self.player2_score > self.player1_score:
            return -1
        else:
            return 0
    
    def get_outcome(self, perspective: int = 1) -> int:
        """
        Get the game outcome from a player's perspective.
        
        Args:
            perspective: 1 for player 1, -1 for player 2
            
        Returns:
            1 if perspective player wins, -1 if loses, 0 if tie
        """
        winner = self.get_winner()
        if winner == 0:
            return 0
        return 1 if winner == perspective else -1
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get a flattened state vector representation of the game.
        Suitable for neural network input.
        
        Returns:
            1D numpy array representing the complete game state
        """
        # Flatten all components
        h_edges = self.horizontal_edges.flatten().astype(float)
        v_edges = self.vertical_edges.flatten().astype(float)
        boxes = self.boxes.flatten().astype(float)
        
        # Add current player as a feature
        current_player_feature = np.array([self.current_player], dtype=float)
        
        # Concatenate all features
        state = np.concatenate([h_edges, v_edges, boxes, current_player_feature])
        
        return state
    
    def get_state_size(self) -> int:
        """Get the size of the state vector."""
        return (self.dots * self.grid_size +  # horizontal edges
                self.grid_size * self.dots +    # vertical edges
                self.grid_size * self.grid_size + # boxes
                1)  # current player
    
    def display(self) -> str:
        """
        Create a text representation of the game board.
        
        Returns:
            String representation of the board
        """
        lines = []
        
        for row in range(self.dots):
            # Draw dots and horizontal edges
            line = ""
            for col in range(self.dots):
                line += "●"
                if col < self.grid_size:
                    if self.horizontal_edges[row][col]:
                        line += "───"
                    else:
                        line += "   "
            lines.append(line)
            
            # Draw vertical edges and boxes
            if row < self.grid_size:
                line = ""
                for col in range(self.dots):
                    if self.vertical_edges[row][col]:
                        line += "│"
                    else:
                        line += " "
                    
                    if col < self.grid_size:
                        box_val = self.boxes[row][col]
                        if box_val == 1:
                            line += " 1 "
                        elif box_val == -1:
                            line += " 2 "
                        else:
                            line += "   "
                lines.append(line)
        
        # Add score information
        lines.append("")
        lines.append(f"Player 1: {self.player1_score} | Player 2: {self.player2_score}")
        lines.append(f"Current player: {1 if self.current_player == 1 else 2}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.display()


def test_game():
    """Simple test function to verify game logic."""
    print("Testing Dots and Boxes Game Implementation\n")
    
    game = DotsAndBoxes(grid_size=2)
    print("Initial state:")
    print(game.display())
    print(f"\nValid moves: {len(game.get_valid_moves())}")
    print(f"State vector size: {game.get_state_size()}")
    
    # Play a few moves
    moves = [
        ('h', 0, 0),  # Top edge of box (0,0)
        ('h', 1, 0),  # Bottom edge of box (0,0)
        ('v', 0, 0),  # Left edge of box (0,0)
        ('v', 0, 1),  # Right edge of box (0,0) - completes box!
    ]
    
    print("\n\nPlaying moves:")
    for move in moves:
        print(f"\nMove: {move}")
        got_another = game.make_move(move)
        print(game.display())
        print(f"Got another turn: {got_another}")
        if game.is_game_over():
            print(f"Game over! Winner: {game.get_winner()}")
            break
    

if __name__ == "__main__":
    test_game()
