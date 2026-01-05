
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our implementations
from dots_and_boxes import DotsAndBoxes
from mlp_evaluator import MLPEvaluator
from train_loop import TrainingLoop
from adaptive_strategy import CuriosityDrivenStrategy, PrecisionFirstStrategy, ConstantStrategy

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Environment Ready")

# --- Strategy Definition ---
strategies = {
    'CuriosityDriven': CuriosityDrivenStrategy(L_init=1, L_max=4, K_init=1, K_max=8, step_iterations=5),
    'PrecisionFirst': PrecisionFirstStrategy(L_start=3, K_start=2, K_max=6, step_iterations=5),
    'Baseline (Constant)': ConstantStrategy(L=2, K=4)
}

# --- Training & Self-Play ---
GRID_SIZE = 3
NUM_ITERATIONS = 5 
GAMES_PER_ITERATION = 5
EPOCHS = 2

results = {}

for name, strategy in strategies.items():
    print(f"\n{'='*40}\nTraining {name}...\n{'='*40}")
    
    trainer = TrainingLoop(grid_size=GRID_SIZE, hidden_sizes=[128, 64])
    
    for t in range(NUM_ITERATIONS):
        L, K = strategy.get_params(t)
        # verbose=True to see progress
        stats = trainer.train_iteration(L, K, GAMES_PER_ITERATION, EPOCHS, verbose=True)
        
    results[name] = trainer

# --- Analysis & Plotting ---

# 1. Training Loss
plt.figure(figsize=(12, 6))
for name, trainer in results.items():
    losses = [h['loss'] for h in trainer.training_history]
    plt.plot(losses, label=name)
plt.title('Training Loss per Strategy')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('new_strategies_loss.png')
print("\nSaved new_strategies_loss.png")

# 2. Benchmark Performance (Win Rate vs Greedy)
plt.figure(figsize=(12, 6))
for name, trainer in results.items():
    # Extract win rate vs Greedy from benchmark results
    # benchmark_results is a list of dicts, one every 5 iterations
    # We need to map them back to iterations
    
    iterations = []
    win_rates = []
    
    for i, stats in enumerate(trainer.benchmark_results):
        # Each entry corresponds to an iteration multiple of 5
        iter_num = (i + 1) * 5
        if 'Greedy' in stats:
            iterations.append(iter_num)
            win_rates.append(stats['Greedy']['win_rate'])
            
    plt.plot(iterations, win_rates, label=f"{name} vs Greedy", marker='o')

plt.title('Win Rate vs Greedy Agent')
plt.xlabel('Iteration')
plt.ylabel('Win Rate')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.savefig('benchmark_performance.png')
print("Saved benchmark_performance.png")

# 3. Signal Stability
plt.figure(figsize=(12, 6))
for name, trainer in results.items():
    stability = trainer.signal_stability
    plt.plot(stability, label=name)
plt.title('Signal Stability (Prediction Variance)')
plt.xlabel('Iteration')
plt.ylabel('Variance')
plt.legend()
plt.savefig('signal_stability.png')
print("Saved signal_stability.png")

print("\nExperiments Complete.")
