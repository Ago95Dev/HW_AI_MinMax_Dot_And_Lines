"""
Quick Test Script - Verify all components work correctly
Run this to ensure everything is installed and working.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from tqdm import tqdm
        print("✓ All external libraries imported successfully\n")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def test_components():
    """Test all project components."""
    print("Testing project components...")
    
    try:
        # Test game
        from dots_and_boxes import DotsAndBoxes
        game = DotsAndBoxes(grid_size=2)
        assert len(game.get_valid_moves()) == 12  # 2x2 grid has 12 edges
        print("✓ Game engine working")
        
        # Test MLP
        from mlp_evaluator import MLPEvaluator
        mlp = MLPEvaluator.create_from_game(game, hidden_sizes=[32, 16])
        state = game.get_state_vector()
        value = mlp.evaluate_state(state)
        assert -1 <= value <= 1
        print("✓ MLP evaluator working")
        
        # Test MinMax
        from minmax import MinMaxAgent
        agent = MinMaxAgent(mlp, L=2, K=3)
        move = agent.select_move(game)
        assert move in game.get_valid_moves()
        print("✓ MinMax agent working")
        
        # Test Training Loop
        from train_loop import TrainingLoop
        trainer = TrainingLoop(grid_size=2, hidden_sizes=[32, 16])
        states, outcome = trainer.play_game(L=1, K=3, verbose=False)
        assert len(states) > 0
        assert outcome in [-1, 0, 1]
        print("✓ Training loop working")
        
        # Test Strategies
        from adaptive_strategy import get_all_strategies
        strategies = get_all_strategies()
        assert len(strategies) == 6
        for name, strategy in strategies.items():
            L, K = strategy.get_params(0)
            assert L > 0 and K > 0
        print("✓ Adaptive strategies working")
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("="*50)
        print("\nYou can now run:")
        print("  - jupyter notebook experiment.ipynb")
        print("  - python <component>.py (for individual tests)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("="*50)
    print("HOMEWORK 1 - COMPONENT TEST")
    print("="*50)
    print()
    
    if not test_imports():
        return False
    
    if not test_components():
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
