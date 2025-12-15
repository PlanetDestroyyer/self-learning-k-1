"""
Metrics tracking for K-1 Self-Learning System.
"""

import numpy as np
from typing import Dict, List
from collections import deque
import json
from pathlib import Path


class MetricsTracker:
    """Tracks and stores training metrics."""

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker.

        Args:
            window_size: Window for moving averages
        """
        self.window_size = window_size

        # Loss tracking
        self.losses = deque(maxlen=window_size)
        self.perplexities = deque(maxlen=window_size)

        # Agent tracking
        self.agent_counts = []
        self.avg_trusts = []

        # Full history for plotting
        self.history: List[Dict] = []

    def record(
        self,
        iteration: int,
        loss: float,
        perplexity: float = None,
        agent_count: int = None,
        avg_trust: float = None,
        **kwargs
    ):
        """
        Record metrics for an iteration.

        Args:
            iteration: Current iteration
            loss: Training loss
            perplexity: Perplexity (if applicable)
            agent_count: Number of agents
            avg_trust: Average trust score
            **kwargs: Additional metrics
        """
        self.losses.append(loss)

        if perplexity:
            self.perplexities.append(perplexity)

        entry = {
            'iteration': iteration,
            'loss': loss,
            'perplexity': perplexity,
            'agent_count': agent_count,
            'avg_trust': avg_trust,
            **kwargs
        }

        self.history.append(entry)

        if agent_count:
            self.agent_counts.append(agent_count)
        if avg_trust:
            self.avg_trusts.append(avg_trust)

    def get_current(self) -> Dict:
        """Get current metrics (moving averages)."""
        return {
            'loss': float(np.mean(self.losses)) if self.losses else 0,
            'perplexity': float(np.mean(self.perplexities)) if self.perplexities else 0,
            'loss_std': float(np.std(self.losses)) if self.losses else 0,
            'agent_count': self.agent_counts[-1] if self.agent_counts else 0,
            'avg_trust': self.avg_trusts[-1] if self.avg_trusts else 0
        }

    def get_best(self) -> Dict:
        """Get best metrics achieved."""
        if not self.history:
            return {}

        losses = [h['loss'] for h in self.history if h['loss']]
        perplexities = [h['perplexity'] for h in self.history if h.get('perplexity')]

        return {
            'best_loss': min(losses) if losses else float('inf'),
            'best_perplexity': min(perplexities) if perplexities else float('inf')
        }

    def get_improvement_rate(self, window: int = 5000) -> float:
        """Get improvement rate over recent window."""
        if len(self.losses) < window:
            return 0.0

        recent = list(self.losses)
        start = np.mean(recent[:window//2])
        end = np.mean(recent[-window//2:])

        return (start - end) / (start + 1e-10)

    def save(self, path: str):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str):
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            self.history = json.load(f)

    def summary(self) -> str:
        """Get summary string."""
        current = self.get_current()
        best = self.get_best()

        return f"""
Metrics Summary:
  Current Loss: {current['loss']:.4f} (std: {current['loss_std']:.4f})
  Current Perplexity: {current['perplexity']:.2f}
  Best Loss: {best.get('best_loss', 'N/A')}
  Best Perplexity: {best.get('best_perplexity', 'N/A')}
  Agent Count: {current['agent_count']}
  Avg Trust: {current['avg_trust']:.3f}
"""
