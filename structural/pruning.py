"""
Self-pruning system for K-1 Self-Learning System.

Removes unused or low-performing agents to maintain efficiency.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from core.agent import Agent
from core.hierarchy import Hierarchy
from core.trust import TrustSystem


class PruningSystem:
    """
    Manages autonomous agent pruning.

    Agents are pruned when they meet ALL criteria:
    - Low trust score
    - Low usage count
    - Dormant for too long
    - Not the only agent in their domain
    - Trust not trending upward
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        trust_threshold: float = 0.2,
        usage_threshold: int = 50,
        dormancy_threshold: int = 10000,
        min_agents_per_manager: int = 3,
        validation_drop_threshold: float = 0.05
    ):
        """
        Initialize pruning system.

        Args:
            hierarchy: Hierarchy to prune
            trust_threshold: Minimum trust to keep agent
            usage_threshold: Minimum usage in window
            dormancy_threshold: Max iterations without use
            min_agents_per_manager: Minimum children per manager
            validation_drop_threshold: Max allowed performance drop
        """
        self.hierarchy = hierarchy
        self.trust_threshold = trust_threshold
        self.usage_threshold = usage_threshold
        self.dormancy_threshold = dormancy_threshold
        self.min_agents_per_manager = min_agents_per_manager
        self.validation_drop_threshold = validation_drop_threshold

        self.pruning_history = []
        self.total_pruned = 0

    def should_prune(self, agent: Agent, current_iteration: int) -> Tuple[bool, str]:
        """
        Determine if an agent should be pruned.

        Args:
            agent: Agent to evaluate
            current_iteration: Current training iteration

        Returns:
            (should_prune, reason)
        """
        # Never prune root
        if agent == self.hierarchy.root:
            return False, "is root"

        # Never prune protected agents
        agent.check_protection(current_iteration)
        if agent.protected:
            return False, "protected"

        # Check if only child
        if agent.is_only_child():
            return False, "only child"

        # Check minimum agents per manager
        if agent.parent and len(agent.parent.children) <= self.min_agents_per_manager:
            return False, "minimum agents"

        # Check if trust is increasing (give benefit of doubt)
        if agent.is_trust_increasing():
            return False, "trust increasing"

        # Now check pruning criteria
        criteria_met = []

        # Low trust
        if agent.trust < self.trust_threshold:
            criteria_met.append("low_trust")

        # Low usage
        if agent.usage_count_window < self.usage_threshold:
            criteria_met.append("low_usage")

        # Dormant
        if (current_iteration - agent.last_used) > self.dormancy_threshold:
            criteria_met.append("dormant")

        # Need at least 2 criteria to prune
        if len(criteria_met) >= 2:
            return True, ", ".join(criteria_met)

        return False, "criteria not met"

    def find_candidates(self, current_iteration: int) -> List[Tuple[Agent, str]]:
        """
        Find all pruning candidates.

        Args:
            current_iteration: Current iteration

        Returns:
            List of (agent, reason) tuples
        """
        candidates = []

        for agent in self.hierarchy.get_all_agents():
            should, reason = self.should_prune(agent, current_iteration)
            if should:
                candidates.append((agent, reason))

        # Sort by trust (prune lowest trust first)
        candidates.sort(key=lambda x: x[0].trust)

        return candidates

    def prune_agent(self, agent: Agent, reason: str, current_iteration: int, trust_cache=None):
        """
        Execute pruning of an agent.

        Args:
            agent: Agent to prune
            reason: Reason for pruning
            current_iteration: Current iteration
            trust_cache: Trust cache to update
        """
        # Remove from trust cache if present
        if trust_cache:
            trust_cache.remove_from_cache(agent.id)

        # Remove from hierarchy (children reassigned automatically)
        self.hierarchy.remove_agent(agent)

        # Record
        self.pruning_history.append({
            'iteration': current_iteration,
            'agent_id': agent.id,
            'specialty': agent.specialty,
            'trust': agent.trust,
            'reason': reason
        })
        self.total_pruned += 1

    def execute_pruning(
        self,
        current_iteration: int,
        trust_cache=None,
        validation_func: Optional[Callable] = None,
        max_prune: int = 5
    ) -> dict:
        """
        Execute pruning cycle.

        Args:
            current_iteration: Current iteration
            trust_cache: Trust cache to update
            validation_func: Optional function to validate pruning
            max_prune: Maximum agents to prune in one cycle

        Returns:
            Dictionary with pruning statistics
        """
        candidates = self.find_candidates(current_iteration)

        pruned = []
        protected = []
        failed_validation = []

        for agent, reason in candidates[:max_prune]:
            # Validate if function provided
            if validation_func:
                baseline_perf = validation_func()
                # Temporarily remove
                self.hierarchy.remove_agent(agent)
                new_perf = validation_func()
                # Restore
                if agent.parent:
                    self.hierarchy.add_agent(agent, agent.parent)

                perf_drop = (baseline_perf - new_perf) / (baseline_perf + 1e-10)

                if perf_drop > self.validation_drop_threshold:
                    agent.mark_protected(current_iteration + 10000)
                    failed_validation.append(agent.id)
                    continue

            # Execute pruning
            self.prune_agent(agent, reason, current_iteration, trust_cache)
            pruned.append(agent.id)

        return {
            'candidates': len(candidates),
            'pruned': len(pruned),
            'protected': len(protected),
            'failed_validation': len(failed_validation),
            'pruned_ids': pruned
        }

    def update_thresholds(
        self,
        trust_threshold: float = None,
        usage_threshold: int = None,
        dormancy_threshold: int = None
    ):
        """Update pruning thresholds."""
        if trust_threshold is not None:
            self.trust_threshold = max(0.05, min(0.5, trust_threshold))
        if usage_threshold is not None:
            self.usage_threshold = max(10, min(500, usage_threshold))
        if dormancy_threshold is not None:
            self.dormancy_threshold = max(1000, min(50000, dormancy_threshold))

    def get_statistics(self) -> dict:
        """Get pruning statistics."""
        return {
            'total_pruned': self.total_pruned,
            'recent_prunings': len([p for p in self.pruning_history[-100:]]),
            'avg_trust_at_pruning': np.mean([p['trust'] for p in self.pruning_history]) if self.pruning_history else 0
        }
