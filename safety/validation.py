"""
Validation system for K-1 Self-Learning System.
"""

import numpy as np
from typing import Callable, Optional


class ValidationSystem:
    """
    Validates structural changes don't hurt performance.
    """

    def __init__(
        self,
        performance_drop_threshold: float = 0.1,
        validation_samples: int = 100
    ):
        """
        Initialize validation system.

        Args:
            performance_drop_threshold: Max allowed performance drop
            validation_samples: Number of samples for validation
        """
        self.performance_drop_threshold = performance_drop_threshold
        self.validation_samples = validation_samples

        self.baseline_performance = None
        self.validation_history = []

    def set_baseline(self, performance: float):
        """Set baseline performance."""
        self.baseline_performance = performance

    def validate_change(
        self,
        current_performance: float,
        change_description: str
    ) -> bool:
        """
        Validate that a change doesn't hurt performance.

        Args:
            current_performance: Performance after change
            change_description: Description of change

        Returns:
            True if change is acceptable
        """
        if self.baseline_performance is None:
            return True

        # Calculate drop (assuming lower is better, like loss/perplexity)
        if current_performance > self.baseline_performance:
            drop = (current_performance - self.baseline_performance) / (self.baseline_performance + 1e-10)
        else:
            drop = 0  # Improved

        is_valid = drop <= self.performance_drop_threshold

        self.validation_history.append({
            'change': change_description,
            'baseline': self.baseline_performance,
            'after': current_performance,
            'drop': drop,
            'accepted': is_valid
        })

        return is_valid

    def update_threshold(self, new_threshold: float):
        """Update performance drop threshold."""
        self.performance_drop_threshold = max(0.01, min(0.5, new_threshold))

    def get_statistics(self) -> dict:
        """Get validation statistics."""
        if not self.validation_history:
            return {'total_validations': 0}

        accepted = sum(1 for v in self.validation_history if v['accepted'])

        return {
            'total_validations': len(self.validation_history),
            'accepted': accepted,
            'rejected': len(self.validation_history) - accepted,
            'acceptance_rate': accepted / len(self.validation_history)
        }
