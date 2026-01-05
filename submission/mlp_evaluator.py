"""
MLP Evaluator (Htrue) - Neural Network for Position Evaluation
Implements a Multi-Layer Perceptron that learns to predict game outcomes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import os


class MLPEvaluator(nn.Module):
    """
    Multi-Layer Perceptron for evaluating game positions.
    
    Input: Flattened game state vector
    Output: Predicted game outcome in [-1, +1]
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64]):
        """
        Initialize the MLP.
        
        Args:
            input_size: Size of the input state vector
            hidden_sizes: List of hidden layer sizes
        """
        super(MLPEvaluator, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Prevent overfitting
            prev_size = hidden_size
        
        # Output layer: single neuron with tanh activation
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())  # Output in [-1, +1]
        
        self.network = nn.Sequential(*layers)
        
        # Training setup
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training statistics
        self.training_losses = []
        self.training_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (input_size,)
            
        Returns:
            Output tensor of shape (batch_size, 1) or (1,)
        """
        return self.network(x)
    
    def evaluate_state(self, state: np.ndarray) -> float:
        """
        Evaluate a single game state.
        
        Args:
            state: Game state as numpy array
            
        Returns:
            Predicted outcome value in [-1, +1]
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            output = self.forward(state_tensor)
            return output.item()
    
    def evaluate_states_batch(self, states: List[np.ndarray]) -> List[float]:
        """
        Evaluate a batch of states.
        
        Args:
            states: List of game states
            
        Returns:
            List of predicted outcome values
        """
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(states))
            outputs = self.forward(states_tensor)
            return outputs.squeeze().tolist()
    
    def train_on_batch(self, states: List[np.ndarray], outcomes: List[int]) -> float:
        """
        Train the network on a batch of (state, outcome) pairs.
        
        Args:
            states: List of game states
            outcomes: List of corresponding outcomes (1, 0, or -1)
            
        Returns:
            Average loss for this batch
        """
        self.train()  # Set to training mode
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        outcomes_tensor = torch.FloatTensor(outcomes).unsqueeze(1)  # Shape: (batch_size, 1)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.forward(states_tensor)
        
        # Compute loss
        loss = self.criterion(predictions, outcomes_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        # Record statistics
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        self.training_count += 1
        
        return loss_value
    
    def train_on_game(self, states: List[np.ndarray], outcome: int, epochs: int = 1) -> float:
        """
        Train on a single game's data.
        
        All states from the game get the same target (the final outcome).
        
        Args:
            states: All states visited during the game
            outcome: Final outcome of the game (1, 0, or -1)
            epochs: Number of times to pass through the data
            
        Returns:
            Average loss over all epochs
        """
        outcomes = [outcome] * len(states)
        
        total_loss = 0.0
        for _ in range(epochs):
            loss = self.train_on_batch(states, outcomes)
            total_loss += loss
        
        return total_loss / epochs
    
    def train_on_games(self, games_data: List[Tuple[List[np.ndarray], int]], 
                       epochs: int = 1) -> float:
        """
        Train on multiple games.
        
        Args:
            games_data: List of (states, outcome) tuples
            epochs: Number of epochs to train
            
        Returns:
            Average loss
        """
        # Flatten all games into one big batch
        all_states = []
        all_outcomes = []
        
        for states, outcome in games_data:
            all_states.extend(states)
            all_outcomes.extend([outcome] * len(states))
        
        if len(all_states) == 0:
            return 0.0
        
        total_loss = 0.0
        for _ in range(epochs):
            loss = self.train_on_batch(all_states, all_outcomes)
            total_loss += loss
        
        return total_loss / epochs
    
    def get_average_recent_loss(self, n: int = 100) -> float:
        """Get average loss over last n training batches."""
        if len(self.training_losses) == 0:
            return 0.0
        recent = self.training_losses[-n:]
        return sum(recent) / len(recent)
    
    def get_prediction_variance(self, states: List[np.ndarray]) -> float:
        """
        Calculate variance of predictions across a set of states.
        Can be used to measure confidence.
        
        Args:
            states: List of states to evaluate
            
        Returns:
            Variance of predictions
        """
        if len(states) == 0:
            return 0.0
        
        predictions = self.evaluate_states_batch(states)
        return np.var(predictions)
    
    def save_model(self, filepath: str):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'training_count': self.training_count,
            'training_losses': self.training_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights and configuration."""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_count = checkpoint['training_count']
        self.training_losses = checkpoint['training_losses']
    
    @staticmethod
    def create_from_game(game, hidden_sizes: List[int] = [128, 64]) -> 'MLPEvaluator':
        """
        Create an MLP evaluator configured for a specific game.
        
        Args:
            game: DotsAndBoxes game instance
            hidden_sizes: Hidden layer sizes
            
        Returns:
            Configured MLPEvaluator
        """
        input_size = game.get_state_size()
        return MLPEvaluator(input_size, hidden_sizes)


def test_mlp():
    """Test function for the MLP evaluator."""
    from dots_and_boxes import DotsAndBoxes
    
    print("Testing MLP Evaluator\n")
    
    # Create game and MLP
    game = DotsAndBoxes(grid_size=2)
    mlp = MLPEvaluator.create_from_game(game, hidden_sizes=[64, 32])
    
    print(f"MLP created with input size: {mlp.input_size}")
    print(f"Hidden layers: {mlp.hidden_sizes}")
    print(f"Total parameters: {sum(p.numel() for p in mlp.parameters())}")
    
    # Test evaluation
    state = game.get_state_vector()
    print(f"\nInitial state shape: {state.shape}")
    
    value = mlp.evaluate_state(state)
    print(f"Initial evaluation (random weights): {value:.4f}")
    
    # Test training on dummy data
    print("\n--- Training Test ---")
    states = [game.get_state_vector() for _ in range(10)]
    outcomes = [1] * 5 + [-1] * 5  # 5 wins, 5 losses
    
    print("Training on 10 random positions...")
    for epoch in range(5):
        loss = mlp.train_on_batch(states, outcomes)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")
    
    # Test evaluation after training
    value_after = mlp.evaluate_state(state)
    print(f"\nEvaluation after training: {value_after:.4f}")
    
    # Test save/load
    print("\n--- Save/Load Test ---")
    filepath = "test_mlp.pth"
    mlp.save_model(filepath)
    print(f"Model saved to {filepath}")
    
    # Create new model and load
    mlp2 = MLPEvaluator.create_from_game(game, hidden_sizes=[64, 32])
    mlp2.load_model(filepath)
    print("Model loaded")
    
    value_loaded = mlp2.evaluate_state(state)
    print(f"Evaluation from loaded model: {value_loaded:.4f}")
    print(f"Match: {abs(value_after - value_loaded) < 1e-6}")
    
    # Cleanup
    os.remove(filepath)
    print("\nTest file cleaned up")


if __name__ == "__main__":
    test_mlp()
