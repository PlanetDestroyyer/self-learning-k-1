"""
Autonomous parameter controller for K-1 Self-Learning System.

Manages Phase 2 autonomous parameter adjustment.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class SystemState:
    """Current system state for parameter decisions."""
    iteration: int
    loss: float
    perplexity: float
    total_agents: int
    active_agents_pct: float
    avg_trust: float
    trust_variance: float
    cache_hit_rate: float
    rollback_count: int
    iterations_since_improvement: int
    structural_changes: int
    avg_error_magnitude: float


class ParameterController:
    """
    Manages autonomous parameter adjustment in Phase 2.

    Uses rule-based heuristics to adapt:
    - Learning rate
    - Top-K agents
    - Exploration rate
    - Pruning thresholds
    - Merging thresholds
    - Operation intervals
    """

    def __init__(self, config: Dict):
        """
        Initialize with Phase 1 configuration.

        Args:
            config: Configuration dictionary
        """
        # Learning parameters
        self.learning_rate = config.get('learning', {}).get('learning_rate', 0.0001)
        self.top_k = config.get('learning', {}).get('top_k', 3)

        # Exploration
        self.exploration_rate = config.get('exploration', {}).get('initial_rate', 0.1)

        # Pruning
        self.prune_interval = config.get('operations', {}).get('prune_interval', 5000)
        self.prune_trust_threshold = config.get('pruning', {}).get('trust_threshold', 0.2)

        # Merging
        self.merge_interval = config.get('operations', {}).get('merge_interval', 5000)
        self.merge_similarity_threshold = config.get('merging', {}).get('similarity_threshold', 0.85)

        # Growing
        self.grow_interval = config.get('operations', {}).get('grow_interval', 2000)

        # Reorganization
        self.reorganize_interval = config.get('operations', {}).get('reorganize_interval', 10000)

        # Cache
        self.cache_threshold = config.get('trust', {}).get('cache_threshold', 0.7)

        # Phase 2 state
        self.phase_2_active = False
        self.adjustment_history: List[Dict] = []

        # Parameter bounds
        self.bounds = {
            'learning_rate': (1e-6, 0.01),
            'top_k': (1, 10),
            'exploration_rate': (0.01, 0.5),
            'prune_trust_threshold': (0.1, 0.4),
            'merge_similarity_threshold': (0.6, 0.95),
            'cache_threshold': (0.5, 0.9),
        }

    def activate_phase_2(self):
        """Activate Phase 2 autonomous adjustment."""
        self.phase_2_active = True
        print("Parameter Controller: Phase 2 ACTIVATED - Autonomous adjustment enabled")

    def check_and_adjust(self, iteration: int, state: SystemState):
        """
        Check system state and adjust parameters.

        Args:
            iteration: Current iteration
            state: Current system state
        """
        if not self.phase_2_active:
            return

        # Only adjust every 1000 iterations
        if iteration % 1000 != 0:
            return

        adjustments = []

        # Rule 1: Learning rate adjustment
        lr_adj = self._adjust_learning_rate(state)
        if lr_adj:
            adjustments.append(lr_adj)

        # Rule 2: Top-K adjustment
        topk_adj = self._adjust_top_k(state)
        if topk_adj:
            adjustments.append(topk_adj)

        # Rule 3: Exploration rate adjustment
        exp_adj = self._adjust_exploration(state)
        if exp_adj:
            adjustments.append(exp_adj)

        # Rule 4: Pruning threshold adjustment
        prune_adj = self._adjust_pruning(state)
        if prune_adj:
            adjustments.append(prune_adj)

        # Rule 5: Merging threshold adjustment
        merge_adj = self._adjust_merging(state)
        if merge_adj:
            adjustments.append(merge_adj)

        # Log adjustments
        if adjustments:
            self.adjustment_history.append({
                'iteration': iteration,
                'adjustments': adjustments
            })
            self._print_adjustments(iteration, adjustments)

    def _adjust_learning_rate(self, state: SystemState) -> Dict:
        """Adjust learning rate based on loss trend."""
        # If long plateau, try increasing LR
        if state.iterations_since_improvement > 10000:
            new_lr = min(self.bounds['learning_rate'][1],
                        self.learning_rate * 1.5)
            if new_lr != self.learning_rate:
                old = self.learning_rate
                self.learning_rate = new_lr
                return {'param': 'learning_rate', 'old': old, 'new': new_lr,
                       'reason': 'long plateau'}

        # If loss unstable (high variance), decrease LR
        if state.avg_error_magnitude > 2.0:
            new_lr = max(self.bounds['learning_rate'][0],
                        self.learning_rate * 0.5)
            if new_lr != self.learning_rate:
                old = self.learning_rate
                self.learning_rate = new_lr
                return {'param': 'learning_rate', 'old': old, 'new': new_lr,
                       'reason': 'high error magnitude'}

        return None

    def _adjust_top_k(self, state: SystemState) -> Dict:
        """Adjust number of agents to update."""
        # If errors complex, update more agents
        if state.avg_error_magnitude > 1.5 and self.top_k < 7:
            old = self.top_k
            self.top_k = min(self.bounds['top_k'][1], self.top_k + 1)
            return {'param': 'top_k', 'old': old, 'new': self.top_k,
                   'reason': 'high error complexity'}

        # If plateau, try more agents
        if state.iterations_since_improvement > 15000 and self.top_k < 8:
            old = self.top_k
            self.top_k = min(self.bounds['top_k'][1], self.top_k + 2)
            return {'param': 'top_k', 'old': old, 'new': self.top_k,
                   'reason': 'plateau exploration'}

        return None

    def _adjust_exploration(self, state: SystemState) -> Dict:
        """Adjust exploration rate."""
        # Plateau -> more exploration
        if state.iterations_since_improvement > 20000:
            new_rate = min(self.bounds['exploration_rate'][1],
                          self.exploration_rate + 0.1)
            if new_rate != self.exploration_rate:
                old = self.exploration_rate
                self.exploration_rate = new_rate
                return {'param': 'exploration_rate', 'old': old, 'new': new_rate,
                       'reason': 'long plateau'}

        # Good progress -> exploit more
        if state.iterations_since_improvement < 1000:
            new_rate = max(self.bounds['exploration_rate'][0],
                          self.exploration_rate - 0.02)
            if new_rate != self.exploration_rate:
                old = self.exploration_rate
                self.exploration_rate = new_rate
                return {'param': 'exploration_rate', 'old': old, 'new': new_rate,
                       'reason': 'exploiting progress'}

        return None

    def _adjust_pruning(self, state: SystemState) -> Dict:
        """Adjust pruning threshold."""
        # Too many rollbacks -> more conservative
        if state.rollback_count > 2:
            new_thresh = min(self.bounds['prune_trust_threshold'][1],
                            self.prune_trust_threshold + 0.05)
            if new_thresh != self.prune_trust_threshold:
                old = self.prune_trust_threshold
                self.prune_trust_threshold = new_thresh
                return {'param': 'prune_trust_threshold', 'old': old, 'new': new_thresh,
                       'reason': 'reducing over-pruning'}

        # Many low-trust agents -> more aggressive
        if state.avg_trust < 0.3 and state.total_agents > 30:
            new_thresh = max(self.bounds['prune_trust_threshold'][0],
                            self.prune_trust_threshold - 0.02)
            if new_thresh != self.prune_trust_threshold:
                old = self.prune_trust_threshold
                self.prune_trust_threshold = new_thresh
                return {'param': 'prune_trust_threshold', 'old': old, 'new': new_thresh,
                       'reason': 'cleaning low-trust agents'}

        return None

    def _adjust_merging(self, state: SystemState) -> Dict:
        """Adjust merging threshold."""
        # Too many agents -> easier merging
        if state.total_agents > 80:
            new_thresh = max(self.bounds['merge_similarity_threshold'][0],
                            self.merge_similarity_threshold - 0.05)
            if new_thresh != self.merge_similarity_threshold:
                old = self.merge_similarity_threshold
                self.merge_similarity_threshold = new_thresh
                return {'param': 'merge_similarity_threshold', 'old': old, 'new': new_thresh,
                       'reason': 'reducing agent count'}

        # Few agents -> preserve diversity
        if state.total_agents < 20:
            new_thresh = min(self.bounds['merge_similarity_threshold'][1],
                            self.merge_similarity_threshold + 0.05)
            if new_thresh != self.merge_similarity_threshold:
                old = self.merge_similarity_threshold
                self.merge_similarity_threshold = new_thresh
                return {'param': 'merge_similarity_threshold', 'old': old, 'new': new_thresh,
                       'reason': 'preserving diversity'}

        return None

    def _print_adjustments(self, iteration: int, adjustments: List[Dict]):
        """Print adjustment summary."""
        print(f"\n{'='*50}")
        print(f"Parameter Adjustments at Iteration {iteration}")
        print(f"{'='*50}")
        for adj in adjustments:
            if isinstance(adj['old'], float):
                print(f"  {adj['param']}: {adj['old']:.6f} -> {adj['new']:.6f}")
            else:
                print(f"  {adj['param']}: {adj['old']} -> {adj['new']}")
            print(f"    Reason: {adj['reason']}")
        print(f"{'='*50}\n")

    def get_current_params(self) -> Dict:
        """Get current parameter values."""
        return {
            'learning_rate': self.learning_rate,
            'top_k': self.top_k,
            'exploration_rate': self.exploration_rate,
            'prune_trust_threshold': self.prune_trust_threshold,
            'merge_similarity_threshold': self.merge_similarity_threshold,
            'cache_threshold': self.cache_threshold,
            'prune_interval': self.prune_interval,
            'merge_interval': self.merge_interval,
            'grow_interval': self.grow_interval,
            'reorganize_interval': self.reorganize_interval
        }

    def get_adjustment_count(self) -> int:
        """Get total number of adjustments made."""
        return sum(len(h['adjustments']) for h in self.adjustment_history)
