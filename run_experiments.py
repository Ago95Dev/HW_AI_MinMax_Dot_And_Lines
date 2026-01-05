
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import warnings
import os
import glob

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our implementations
from dots_and_boxes import DotsAndBoxes
from mlp_evaluator import MLPEvaluator
from minmax import MinMaxAgent, RandomAgent
from train_loop import TrainingLoop
from adaptive_strategy import *

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Environment Ready")

# --- 2. Strategy Definition ---
strategies = {
    'Progressive': ProgressiveDeepeningStrategy(L_init=1, L_max=4, K_constant=5, step_iterations=10),
    'Inverse': InverseRelationshipStrategy(L_init=1, L_max=4, K_init=8, K_min=3, step_iterations=10),
    'Exponential': ExponentialGrowthStrategy(L_init=1, L_max=4, K_init=8, K_min=3, growth_rate=0.05),
    'Constant': ConstantStrategy(L=2, K=5)
}

# Visualize Strategies
max_iter = 40
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for name, strategy in strategies.items():
    iterations = range(max_iter)
    params = [strategy.get_params(i) for i in iterations]
    L_values = [p[0] for p in params]
    K_values = [p[1] for p in params]
    
    ax1.plot(iterations, L_values, label=name, marker='o', markersize=3)
    ax2.plot(iterations, K_values, label=name, marker='s', markersize=3)

ax1.set_title('Depth Cut $L(t)$')
ax1.set_xlabel('Iteration $t$')
ax1.set_ylabel('Depth $L$')
ax1.legend()

ax2.set_title('Width Cut $K(t)$')
ax2.set_xlabel('Iteration $t$')
ax2.set_ylabel('Width $K$')
ax2.legend()

plt.tight_layout()
plt.savefig('strategies_plot.png')
print("Saved strategies_plot.png")

# --- 3. Training & Self-Play ---
GRID_SIZE = 3
NUM_ITERATIONS = 20 # Reduced slightly for speed in this environment, originally 30
GAMES_PER_ITERATION = 5
EPOCHS = 2

results = {}

for name, strategy in strategies.items():
    print(f"\n{'='*40}\nTraining {name}...\n{'='*40}")
    
    trainer = TrainingLoop(grid_size=GRID_SIZE, hidden_sizes=[128, 64])
    history = []
    checkpoints = {}
    
    for t in range(NUM_ITERATIONS):
        L, K = strategy.get_params(t)
        # verbose=False to reduce output
        stats = trainer.train_iteration(L, K, GAMES_PER_ITERATION, EPOCHS, verbose=False)
        history.append(stats)
        
        if (t + 1) % 10 == 0:
            cp_path = f"cp_{name}_{t+1}.pth"
            trainer.save_checkpoint(cp_path)
            checkpoints[t+1] = cp_path
            print(f"Saved checkpoint: {cp_path}")
            
    results[name] = {
        'trainer': trainer,
        'history': history,
        'checkpoints': checkpoints
    }
    
    print(f"Final Loss: {history[-1]['loss']:.4f}")

# Plot Training Loss
plt.figure(figsize=(12, 6))
for name, data in results.items():
    losses = [h['loss'] for h in data['history']]
    plt.plot(losses, label=name)
plt.title('Training Loss per Strategy')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('training_loss_comparison.png')
print("Saved training_loss_comparison.png")

# --- 4. Experiment 1: Generational Battle ---
def play_match(agent1, agent2, games=20):
    wins, ties, losses = 0, 0, 0
    # game = DotsAndBoxes(GRID_SIZE) # Unused variable
    
    for _ in range(games):
        g = DotsAndBoxes(GRID_SIZE)
        while not g.is_game_over():
            if g.current_player == 1:
                g.make_move(agent1.select_move(g))
            else:
                g.make_move(agent2.select_move(g))
        
        w = g.get_winner()
        if w == 1: wins += 1
        elif w == -1: losses += 1
        else: ties += 1
            
    return wins, ties, losses

strat_name = 'Exponential'
data = results[strat_name]
checkpoints = data['checkpoints']

print(f"\nGenerational Battle ({strat_name})\n")
print(f"{'Matchup':<20} | {'Win Rate (New vs Old)':<25}")
print("-" * 50)

sorted_iters = sorted(checkpoints.keys())
if len(sorted_iters) > 1:
    for i in range(len(sorted_iters)-1):
        t_old = sorted_iters[i]
        t_new = sorted_iters[i+1]
        
        mlp_old = MLPEvaluator(DotsAndBoxes(GRID_SIZE).get_state_size())
        mlp_old.load_model(checkpoints[t_old])
        agent_old = MinMaxAgent(mlp_old, L=2, K=4)
        
        mlp_new = MLPEvaluator(DotsAndBoxes(GRID_SIZE).get_state_size())
        mlp_new.load_model(checkpoints[t_new])
        agent_new = MinMaxAgent(mlp_new, L=2, K=4)
        
        w, t, l = play_match(agent_new, agent_old, games=20)
        win_rate = (w / 20) * 100
        
        print(f"Iter {t_new} vs {t_old:<5} | {win_rate:>5.1f}% Wins")
else:
    print("Not enough checkpoints for generational battle (need at least 2).")

# --- 5. Experiment 2: Cost/Benefit Analysis ---
cost_benefit_data = []
rand_agent = RandomAgent()

print("\nCost/Benefit Analysis (vs Random)\n")

for name, data in results.items():
    trainer = data['trainer']
    agent = MinMaxAgent(trainer.mlp, L=3, K=5)
    
    wins = 0
    total_nodes = 0
    games = 10
    
    for _ in range(games):
        g = DotsAndBoxes(GRID_SIZE)
        while not g.is_game_over():
            if g.current_player == 1:
                move = agent.select_move(g)
                stats = agent.get_statistics()
                total_nodes += stats['nodes_evaluated']
            else:
                move = rand_agent.select_move(g)
            g.make_move(move)
        if g.get_winner() == 1: wins += 1
            
    avg_nodes = total_nodes / games
    win_rate = (wins / games) * 100
    
    cost_benefit_data.append({
        'Strategy': name,
        'Win Rate': win_rate,
        'Avg Nodes': avg_nodes
    })

df_cost = pd.DataFrame(cost_benefit_data)
print(df_cost)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cost, x='Avg Nodes', y='Win Rate', hue='Strategy', s=200)
plt.title('Efficiency Frontier: Win Rate vs Computational Cost')
plt.grid(True)
plt.savefig('cost_benefit_analysis.png')
print("Saved cost_benefit_analysis.png")

# --- 6. Hyperparameter Tuning ---
growth_rates = [0.01, 0.05, 0.10]
tuning_results = []

print("\nTuning Exponential Growth Rate...\n")

for rate in growth_rates:
    strat = ExponentialGrowthStrategy(growth_rate=rate)
    trainer = TrainingLoop(GRID_SIZE, [64, 32])
    
    # Train for 10 iterations (reduced for speed)
    for t in range(10):
        L, K = strat.get_params(t)
        trainer.train_iteration(L, K, num_games=3, verbose=False)
        
    final_loss = trainer.mlp.get_average_recent_loss()
    tuning_results.append({'Rate': rate, 'Final Loss': final_loss})
    print(f"Rate {rate}: Loss {final_loss:.4f}")

best_rate = min(tuning_results, key=lambda x: x['Final Loss'])['Rate']
print(f"\nOptimal Growth Rate: {best_rate}")

# Cleanup
for f in glob.glob("cp_*.pth"):
    os.remove(f)
print("Cleanup complete.")
