"""
Adaptive Strategies for L(t) and K(t)
Multiple strategies for evolving search parameters during training.
"""

from typing import Tuple, Callable
import numpy as np
from abc import ABC, abstractmethod


class AdaptiveStrategy(ABC):
    """Base class for adaptive strategies."""
    
    @abstractmethod
    def get_params(self, iteration: int) -> Tuple[int, int]:
        """
        Get L and K parameters for a given iteration.
        
        Args:
            iteration: Current training iteration (0-indexed)
            
        Returns:
            Tuple of (L, K)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get strategy description."""
        pass


class ProgressiveDeepeningStrategy(AdaptiveStrategy):
    """
    Strategy 1: Progressive Deepening
    Gradually increase depth while keeping width constant.
    
    Rationale: As the network learns, allow it to search deeper.
    """
    
    def __init__(self, L_init: int = 1, L_max: int = 5, K_constant: int = 5, 
                 step_iterations: int = 10):
        """
        Args:
            L_init: Initial depth
            L_max: Maximum depth
            K_constant: Constant width
            step_iterations: Iterations before increasing depth
        """
        self.L_init = L_init
        self.L_max = L_max
        self.K_constant = K_constant
        self.step_iterations = step_iterations
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        L = min(self.L_max, self.L_init + (iteration // self.step_iterations))
        K = self.K_constant
        return L, K
    
    def get_name(self) -> str:
        return "Progressive Deepening"
    
    def get_description(self) -> str:
        return f"L: {self.L_init}→{self.L_max} (every {self.step_iterations} iter), K: {self.K_constant} (constant)"


class InverseRelationshipStrategy(AdaptiveStrategy):
    """
    Strategy 2: Inverse Relationship
    Increase depth while decreasing width.
    
    Rationale: As network improves, trust its evaluation more (need fewer alternatives)
    and search deeper (rely on prediction at greater depths).
    """
    
    def __init__(self, L_init: int = 1, L_max: int = 5, 
                 K_init: int = 10, K_min: int = 3,
                 step_iterations: int = 10):
        """
        Args:
            L_init: Initial depth
            L_max: Maximum depth
            K_init: Initial width
            K_min: Minimum width
            step_iterations: Iterations before adjusting
        """
        self.L_init = L_init
        self.L_max = L_max
        self.K_init = K_init
        self.K_min = K_min
        self.step_iterations = step_iterations
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        steps = iteration // self.step_iterations
        L = min(self.L_max, self.L_init + steps)
        K = max(self.K_min, self.K_init - steps)
        return L, K
    
    def get_name(self) -> str:
        return "Inverse Relationship"
    
    def get_description(self) -> str:
        return f"L: {self.L_init}→{self.L_max}, K: {self.K_init}→{self.K_min} (every {self.step_iterations} iter)"


class ExponentialGrowthStrategy(AdaptiveStrategy):
    """
    Strategy 3: Exponential Growth
    L grows exponentially (faster early on), K decreases linearly.
    
    Rationale: Rapid early exploration, then refinement.
    """
    
    def __init__(self, L_init: int = 1, L_max: int = 6,
                 K_init: int = 8, K_min: int = 3,
                 growth_rate: float = 0.1):
        """
        Args:
            L_init: Initial depth
            L_max: Maximum depth
            K_init: Initial width
            K_min: Minimum width
            growth_rate: Exponential growth rate for L
        """
        self.L_init = L_init
        self.L_max = L_max
        self.K_init = K_init
        self.K_min = K_min
        self.growth_rate = growth_rate
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        # Exponential growth for L
        L = int(min(self.L_max, self.L_init + np.floor(self.L_init * (np.exp(self.growth_rate * iteration) - 1))))
        
        # Linear decrease for K
        if iteration == 0:
            K = self.K_init
        else:
            # Decrease K proportionally to L increase
            progress = (L - self.L_init) / (self.L_max - self.L_init)
            K = int(max(self.K_min, self.K_init - progress * (self.K_init - self.K_min)))
        
        return L, K
    
    def get_name(self) -> str:
        return "Exponential Growth"
    
    def get_description(self) -> str:
        return f"L: {self.L_init}→{self.L_max} (exponential), K: {self.K_init}→{self.K_min} (linear)"


class SigmoidStrategy(AdaptiveStrategy):
    """
    Strategy 4: Sigmoid Growth
    Both L and K follow sigmoid curves for smooth transitions.
    
    Rationale: Gradual acceleration in early training, then stabilization.
    """
    
    def __init__(self, L_init: int = 1, L_max: int = 5,
                 K_init: int = 10, K_min: int = 3,
                 midpoint: int = 25, steepness: float = 0.2):
        """
        Args:
            L_init: Initial depth
            L_max: Maximum depth
            K_init: Initial width
            K_min: Minimum width
            midpoint: Iteration at which sigmoid is at midpoint
            steepness: Steepness of sigmoid curve
        """
        self.L_init = L_init
        self.L_max = L_max
        self.K_init = K_init
        self.K_min = K_min
        self.midpoint = midpoint
        self.steepness = steepness
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function normalized to [0, 1]."""
        return 1 / (1 + np.exp(-self.steepness * (x - self.midpoint)))
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        progress = self._sigmoid(iteration)
        
        L = int(self.L_init + progress * (self.L_max - self.L_init))
        K = int(self.K_init - progress * (self.K_init - self.K_min))
        
        return L, K
    
    def get_name(self) -> str:
        return "Sigmoid Growth"
    
    def get_description(self) -> str:
        return f"L: {self.L_init}→{self.L_max}, K: {self.K_init}→{self.K_min} (sigmoid, mid={self.midpoint})"


class ConstantStrategy(AdaptiveStrategy):
    """
    Baseline: Constant parameters (no adaptation).
    """
    
    def __init__(self, L: int = 3, K: int = 5):
        """
        Args:
            L: Constant depth
            K: Constant width
        """
        self.L = L
        self.K = K
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        return self.L, self.K
    
    def get_name(self) -> str:
        return "Constant"
    
    def get_description(self) -> str:
        return f"L: {self.L} (constant), K: {self.K} (constant)"


class StaircaseStrategy(AdaptiveStrategy):
    """
    Strategy 5: Staircase
    Discrete jumps at specific intervals.
    
    Rationale: Allow model to stabilize at each level before increasing complexity.
    """
    
    def __init__(self, L_schedule: list = [1, 2, 3, 4, 5],
                 K_schedule: list = [10, 8, 6, 5, 4],
                 iterations_per_step: int = 15):
        """
        Args:
            L_schedule: List of L values for each step
            K_schedule: List of K values for each step
            iterations_per_step: Iterations at each step before jumping
        """
        assert len(L_schedule) == len(K_schedule), "L and K schedules must have same length"
        self.L_schedule = L_schedule
        self.K_schedule = K_schedule
        self.iterations_per_step = iterations_per_step
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        step = min(iteration // self.iterations_per_step, len(self.L_schedule) - 1)
        return self.L_schedule[step], self.K_schedule[step]
    
    def get_name(self) -> str:
        return "Staircase"
    
    def get_description(self) -> str:
        return f"Steps: {len(self.L_schedule)}, {self.iterations_per_step} iter/step"

class CuriosityDrivenStrategy(AdaptiveStrategy):
    """
    Strategy: Curiosity-Driven (Incremental)
    Start with L=1, K=1 to generate data quickly.
    Increase L and K every N iterations.
    """
    
    def __init__(self, L_init: int = 1, L_max: int = 5,
                 K_init: int = 1, K_max: int = 10,
                 step_iterations: int = 5):
        self.L_init = L_init
        self.L_max = L_max
        self.K_init = K_init
        self.K_max = K_max
        self.step_iterations = step_iterations
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        steps = iteration // self.step_iterations
        L = min(self.L_max, self.L_init + steps)
        K = min(self.K_max, self.K_init + steps)
        return L, K
    
    def get_name(self) -> str:
        return "Curiosity Driven"
    
    def get_description(self) -> str:
        return f"L: {self.L_init}→{self.L_max}, K: {self.K_init}→{self.K_max} (increasing every {self.step_iterations} iter)"


class PrecisionFirstStrategy(AdaptiveStrategy):
    """
    Strategy: Precision-First
    Start with high L (depth) but very small K (width).
    Force model to understand long-term consequences of few moves.
    """
    
    def __init__(self, L_start: int = 4, K_start: int = 2,
                 K_max: int = 6, step_iterations: int = 10):
        self.L_start = L_start
        self.K_start = K_start
        self.K_max = K_max
        self.step_iterations = step_iterations
    
    def get_params(self, iteration: int) -> Tuple[int, int]:
        # L stays constant (high)
        L = self.L_start
        
        # K increases slowly to allow more breadth as confidence grows
        steps = iteration // self.step_iterations
        K = min(self.K_max, self.K_start + steps)
        
        return L, K
    
    def get_name(self) -> str:
        return "Precision First"
    
    def get_description(self) -> str:
        return f"L: {self.L_start} (constant), K: {self.K_start}→{self.K_max} (increasing every {self.step_iterations} iter)"


# Convenience function to get all strategies
def get_all_strategies() -> dict:
    """
    Get all available strategies.
    
    Returns:
        Dictionary mapping strategy names to strategy instances
    """
    strategies = {
        'progressive': ProgressiveDeepeningStrategy(),
        'inverse': InverseRelationshipStrategy(),
        'exponential': ExponentialGrowthStrategy(),
        'sigmoid': SigmoidStrategy(),
        'constant': ConstantStrategy(),
        'staircase': StaircaseStrategy(),
        'curiosity': CuriosityDrivenStrategy(),
        'precision': PrecisionFirstStrategy()
    }
    return strategies


def visualize_strategy(strategy: AdaptiveStrategy, max_iterations: int = 50):
    """
    Visualize how a strategy evolves L and K over iterations.
    
    Args:
        strategy: Strategy to visualize
        max_iterations: Number of iterations to plot
    """
    iterations = list(range(max_iterations))
    L_values = []
    K_values = []
    
    for i in iterations:
        L, K = strategy.get_params(i)
        L_values.append(L)
        K_values.append(K)
    
    print(f"\n{strategy.get_name()}")
    print(f"{strategy.get_description()}")
    print("\nIteration | L | K")
    print("-" * 20)
    for i in [0, 10, 20, 30, 40, max_iterations-1]:
        if i < max_iterations:
            print(f"{i:9d} | {L_values[i]:1d} | {K_values[i]:2d}")


def test_strategies():
    """Test all adaptive strategies."""
    print("Testing Adaptive Strategies\n")
    print("="*60)
    
    strategies = get_all_strategies()
    
    for name, strategy in strategies.items():
        visualize_strategy(strategy, max_iterations=50)
        print("="*60)


if __name__ == "__main__":
    test_strategies()
