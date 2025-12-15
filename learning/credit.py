"""
Credit assignment for K-1 Self-Learning System.

Implements trust-based credit assignment - the core K-1 innovation.
Trust decides WHICH agents to update based on responsibility scores.
"""

import numpy as np
from typing import List, Tuple
from core.agent import Agent
from core.trust import TrustSystem


class CreditAssignment:
    """
    Trust-based credit assignment system.

    This is the KEY K-1 innovation:
    - Instead of updating ALL weights (standard backprop)
    - We select TOP-K agents by responsibility × trust
    - Only those agents get updated
    """

    def __init__(
        self,
        trust_system: TrustSystem,
        top_k: int = 3,
        min_activation: float = 0.01
    ):
        """
        Initialize credit assignment.

        Args:
            trust_system: Trust system for scoring
            top_k: Number of top agents to update
            min_activation: Minimum activation to be considered
        """
        self.trust_system = trust_system
        self.top_k = top_k
        self.min_activation = min_activation

        # Statistics
        self.selection_history = []

    def compute_responsibility(
        self,
        agent: Agent,
        error_magnitude: float
    ) -> float:
        """
        Compute responsibility score for an agent.

        Responsibility = activation_level × error_magnitude

        Args:
            agent: Agent to score
            error_magnitude: Current error magnitude

        Returns:
            Responsibility score
        """
        if agent.activation_level < self.min_activation:
            return 0.0

        return agent.activation_level * error_magnitude

    def compute_ranking_score(
        self,
        agent: Agent,
        responsibility: float
    ) -> float:
        """
        Compute ranking score for agent selection.

        Ranking = responsibility × (1 + trust)

        Higher trust agents get priority when responsibility is similar.

        Args:
            agent: Agent to rank
            responsibility: Responsibility score

        Returns:
            Ranking score
        """
        return responsibility * (1.0 + agent.trust)

    def select_agents_for_update(
        self,
        activated_agents: List[Agent],
        error_magnitude: float
    ) -> List[Tuple[Agent, float, float]]:
        """
        Select top-K agents for weight update.

        This is the CORE K-1 mechanism:
        1. Compute responsibility for each activated agent
        2. Rank by responsibility × trust
        3. Return top-K agents

        Args:
            activated_agents: List of agents activated in forward pass
            error_magnitude: Current error magnitude

        Returns:
            List of (agent, responsibility, ranking_score) tuples
        """
        if not activated_agents:
            return []

        # Compute scores for all agents
        scored_agents = []

        for agent in activated_agents:
            responsibility = self.compute_responsibility(agent, error_magnitude)
            ranking = self.compute_ranking_score(agent, responsibility)

            if responsibility > 0:
                scored_agents.append((agent, responsibility, ranking))

        # Sort by ranking score (descending)
        scored_agents.sort(key=lambda x: x[2], reverse=True)

        # Select top-K
        selected = scored_agents[:self.top_k]

        # Record selection
        if selected:
            self.selection_history.append({
                'num_activated': len(activated_agents),
                'num_selected': len(selected),
                'top_trust': selected[0][0].trust if selected else 0,
                'avg_responsibility': np.mean([s[1] for s in selected])
            })

        return selected

    def update_trust_based_on_improvement(
        self,
        selected_agents: List[Tuple[Agent, float, float]],
        old_loss: float,
        new_loss: float
    ):
        """
        Update trust scores based on actual improvement.

        This is CRITICAL - we measure ACTUAL improvement, not fake.

        Args:
            selected_agents: Agents that were updated
            old_loss: Loss before update
            new_loss: Loss after update
        """
        improvement = old_loss - new_loss

        for agent, responsibility, ranking in selected_agents:
            if improvement > 0:
                # Agent helped - increase trust
                self.trust_system.report_success(agent, improvement)
            else:
                # Agent didn't help - decrease trust
                self.trust_system.report_error(agent, abs(improvement))

    def update_top_k(self, new_k: int):
        """Update number of agents to select."""
        self.top_k = max(1, min(20, new_k))

    def get_statistics(self) -> dict:
        """Get credit assignment statistics."""
        if not self.selection_history:
            return {'total_selections': 0}

        recent = self.selection_history[-1000:]

        return {
            'total_selections': len(self.selection_history),
            'avg_selected': np.mean([s['num_selected'] for s in recent]),
            'avg_activated': np.mean([s['num_activated'] for s in recent]),
            'avg_top_trust': np.mean([s['top_trust'] for s in recent]),
            'selection_ratio': np.mean([s['num_selected'] / max(1, s['num_activated']) for s in recent])
        }
