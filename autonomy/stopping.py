"""
Autonomous stopping controller for K-1 Self-Learning System.

Decides when to stop training based on multiple signals.
"""

import numpy as np
from typing import Tuple, Dict, List
from collections import deque


class StoppingController:
    """
    Manages autonomous training termination.

    Monitors:
    - Performance plateau
    - Diminishing returns
    - Goal achievement
    - Iteration limits
    """

    def __init__(
        self,
        max_iterations: int = 50000,
        patience: int = 10000,
        min_improvement: float = 0.001,
        target_perplexity: float = None
    ):
        """
        Initialize stopping controller.

        Args:
            max_iterations: Maximum iterations allowed
            patience: Iterations without improvement before stopping
            min_improvement: Minimum improvement to reset patience
            target_perplexity: Target perplexity to achieve
        """
        self.max_iterations = max_iterations
        self.patience = patience
        self.min_improvement = min_improvement
        self.target_perplexity = target_perplexity

        # Tracking
        self.best_loss = float('inf')
        self.best_perplexity = float('inf')
        self.best_iteration = 0
        self.iterations_since_improvement = 0

        # History
        self.loss_history = deque(maxlen=10000)
        self.perplexity_history = deque(maxlen=10000)

    def update(
        self,
        iteration: int,
        loss: float,
        perplexity: float = None
    ) -> Tuple[bool, str]:
        """
        Update with current metrics and check stopping.

        Args:
            iteration: Current iteration
            loss: Current loss
            perplexity: Current perplexity (optional)

        Returns:
            (should_stop, reason)
        """
        self.loss_history.append(loss)

        if perplexity:
            self.perplexity_history.append(perplexity)

            # Check improvement
            if perplexity < self.best_perplexity - self.min_improvement:
                self.best_perplexity = perplexity
                self.best_iteration = iteration
                self.iterations_since_improvement = 0
            else:
                self.iterations_since_improvement += 1
        else:
            # Use loss for improvement tracking
            if loss < self.best_loss - self.min_improvement:
                self.best_loss = loss
                self.best_iteration = iteration
                self.iterations_since_improvement = 0
            else:
                self.iterations_since_improvement += 1

        # Check stopping conditions
        return self._check_stopping(iteration, loss, perplexity)

    def _check_stopping(
        self,
        iteration: int,
        loss: float,
        perplexity: float
    ) -> Tuple[bool, str]:
        """Check all stopping conditions."""

        # 1. Max iterations
        if iteration >= self.max_iterations:
            return True, f"max iterations reached ({self.max_iterations})"

        # 2. Patience exhausted (plateau)
        if self.iterations_since_improvement >= self.patience:
            return True, f"no improvement for {self.patience} iterations"

        # 3. Target achieved
        if self.target_perplexity and perplexity:
            if perplexity <= self.target_perplexity:
                return True, f"target perplexity {self.target_perplexity} achieved"

        # 4. Diminishing returns (very slow improvement)
        if len(self.loss_history) >= 5000:
            recent = list(self.loss_history)[-5000:]
            improvement = recent[0] - recent[-1]
            improvement_per_1k = improvement / 5.0

            if improvement_per_1k < self.min_improvement / 10:
                # Less than 1/10 of min improvement per 1k iterations
                return True, "diminishing returns"

        # 5. Loss diverging
        if len(self.loss_history) >= 1000:
            recent = list(self.loss_history)[-1000:]
            if np.mean(recent[-100:]) > np.mean(recent[:100]) * 2:
                return True, "loss diverging"

        return False, ""

    def set_target_perplexity(self, target: float):
        """Set target perplexity."""
        self.target_perplexity = target
        print(f"Stopping Controller: Target perplexity set to {target:.2f}")

    def get_statistics(self) -> Dict:
        """Get stopping statistics."""
        return {
            'best_loss': self.best_loss,
            'best_perplexity': self.best_perplexity,
            'best_iteration': self.best_iteration,
            'iterations_since_improvement': self.iterations_since_improvement,
            'patience_remaining': self.patience - self.iterations_since_improvement,
            'history_length': len(self.loss_history)
        }

    def generate_report(self, final_iteration: int, reason: str) -> str:
        """Generate stopping report."""
        report = f"""
{'='*60}
TRAINING STOPPED
{'='*60}

Reason: {reason}
Final Iteration: {final_iteration:,}

Performance:
  Best Loss: {self.best_loss:.4f}
  Best Perplexity: {self.best_perplexity:.2f}
  Best at Iteration: {self.best_iteration:,}

Patience:
  Allowed: {self.patience:,} iterations
  Used: {self.iterations_since_improvement:,} iterations

{'='*60}
"""
        return report
