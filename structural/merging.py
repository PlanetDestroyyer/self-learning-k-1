"""
Self-merging system for K-1 Self-Learning System.

Combines redundant agents to reduce complexity.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from itertools import combinations
from core.agent import Agent
from core.hierarchy import Hierarchy


class MergingSystem:
    """
    Manages autonomous agent merging.

    Agents are merged when:
    - High similarity (coactivation, output correlation)
    - Both have reasonable trust
    - Same parent (siblings)
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        similarity_threshold: float = 0.85,
        min_trust: float = 0.3,
        validation_drop_threshold: float = 0.05
    ):
        """
        Initialize merging system.

        Args:
            hierarchy: Hierarchy to operate on
            similarity_threshold: Minimum similarity to merge
            min_trust: Minimum trust for both agents
            validation_drop_threshold: Max performance drop
        """
        self.hierarchy = hierarchy
        self.similarity_threshold = similarity_threshold
        self.min_trust = min_trust
        self.validation_drop_threshold = validation_drop_threshold

        # Track coactivations
        self.coactivation_counts: Dict[Tuple[str, str], int] = {}
        self.total_activations = 0

        self.merge_history = []
        self.total_merges = 0

    def record_coactivation(self, agent_ids: List[str]):
        """
        Record that agents were activated together.

        Args:
            agent_ids: List of activated agent IDs
        """
        self.total_activations += 1

        # Record all pairs
        for id1, id2 in combinations(sorted(agent_ids), 2):
            key = (id1, id2)
            self.coactivation_counts[key] = self.coactivation_counts.get(key, 0) + 1

    def compute_coactivation_similarity(self, agent_a: Agent, agent_b: Agent) -> float:
        """Compute similarity based on coactivation frequency."""
        if self.total_activations == 0:
            return 0.0

        key = tuple(sorted([agent_a.id, agent_b.id]))
        coactivations = self.coactivation_counts.get(key, 0)

        return coactivations / (self.total_activations + 1e-10)

    def compute_trust_similarity(self, agent_a: Agent, agent_b: Agent) -> float:
        """Compute similarity based on trust scores."""
        return 1.0 - abs(agent_a.trust - agent_b.trust)

    def compute_weight_similarity(self, agent_a: Agent, agent_b: Agent) -> float:
        """Compute similarity based on weight correlation."""
        # Flatten weights
        w_a = np.concatenate([agent_a.weights['W1'].flatten(),
                              agent_a.weights['W2'].flatten()])
        w_b = np.concatenate([agent_b.weights['W1'].flatten(),
                              agent_b.weights['W2'].flatten()])

        # Correlation
        if np.std(w_a) < 1e-10 or np.std(w_b) < 1e-10:
            return 0.0

        corr = np.corrcoef(w_a, w_b)[0, 1]
        return (corr + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

    def compute_similarity(self, agent_a: Agent, agent_b: Agent) -> float:
        """
        Compute overall similarity between two agents.

        Args:
            agent_a: First agent
            agent_b: Second agent

        Returns:
            Similarity score (0 to 1)
        """
        coact_sim = self.compute_coactivation_similarity(agent_a, agent_b)
        trust_sim = self.compute_trust_similarity(agent_a, agent_b)
        weight_sim = self.compute_weight_similarity(agent_a, agent_b)

        # Weighted combination
        return 0.4 * coact_sim + 0.3 * trust_sim + 0.3 * weight_sim

    def find_merge_candidates(self) -> List[Tuple[Agent, Agent, float]]:
        """
        Find pairs of agents that could be merged.

        Returns:
            List of (agent_a, agent_b, similarity) tuples
        """
        candidates = []
        seen_pairs = set()

        # Only consider siblings (same parent)
        for agent in self.hierarchy.get_all_agents():
            if agent.parent is None:
                continue

            # Check trust threshold
            if agent.trust < self.min_trust:
                continue

            siblings = self.hierarchy.get_siblings(agent)

            for sibling in siblings:
                # Skip if already seen this pair
                pair_key = tuple(sorted([agent.id, sibling.id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Check trust threshold
                if sibling.trust < self.min_trust:
                    continue

                # Compute similarity
                similarity = self.compute_similarity(agent, sibling)

                if similarity >= self.similarity_threshold:
                    candidates.append((agent, sibling, similarity))

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def merge_agents(
        self,
        agent_a: Agent,
        agent_b: Agent,
        current_iteration: int
    ) -> Agent:
        """
        Create a merged agent from two agents.

        Args:
            agent_a: First agent
            agent_b: Second agent
            current_iteration: Current iteration

        Returns:
            New merged agent
        """
        # Create merged agent
        merged = Agent(
            agent_id=f"merged_{agent_a.id}_{agent_b.id}_{current_iteration}",
            agent_type=agent_a.agent_type,
            specialty=f"{agent_a.specialty}+{agent_b.specialty}",
            input_dim=agent_a.input_dim,
            hidden_dim=agent_a.hidden_dim,
            output_dim=agent_a.output_dim,
            initial_trust=max(agent_a.trust, agent_b.trust),
            creation_iteration=current_iteration
        )

        # Average weights
        for key in ['W1', 'b1', 'W2', 'b2']:
            merged.weights[key] = 0.5 * agent_a.weights[key] + 0.5 * agent_b.weights[key]

        # Combine statistics
        merged.success_count = agent_a.success_count + agent_b.success_count
        merged.failure_count = agent_a.failure_count + agent_b.failure_count
        merged.total_error_reduction = agent_a.total_error_reduction + agent_b.total_error_reduction
        merged.usage_count = agent_a.usage_count + agent_b.usage_count

        return merged

    def execute_merge(
        self,
        agent_a: Agent,
        agent_b: Agent,
        merged: Agent,
        current_iteration: int
    ):
        """
        Execute the merge operation.

        Args:
            agent_a: First agent to remove
            agent_b: Second agent to remove
            merged: New merged agent
            current_iteration: Current iteration
        """
        parent = agent_a.parent

        # Save children BEFORE removing
        children_a = list(agent_a.children)
        children_b = list(agent_b.children)

        # Remove old agents
        self.hierarchy.remove_agent(agent_a)
        self.hierarchy.remove_agent(agent_b)

        # Add merged agent
        if parent:
            self.hierarchy.add_agent(merged, parent)

        # Transfer children
        for child in children_a + children_b:
            if child.parent is None:  # Only if not already assigned
                merged.add_child(child)

        # Record
        self.merge_history.append({
            'iteration': current_iteration,
            'agent_a': agent_a.id,
            'agent_b': agent_b.id,
            'merged_id': merged.id,
            'similarity': self.compute_similarity(agent_a, agent_b)
        })
        self.total_merges += 1

    def execute_merging_cycle(
        self,
        current_iteration: int,
        max_merges: int = 3
    ) -> dict:
        """
        Execute merging cycle.

        Args:
            current_iteration: Current iteration
            max_merges: Maximum merges per cycle

        Returns:
            Statistics dictionary
        """
        candidates = self.find_merge_candidates()

        merged_pairs = []
        merged_ids = set()

        for agent_a, agent_b, similarity in candidates:
            if len(merged_pairs) >= max_merges:
                break

            # Skip if either agent was already merged this cycle
            if agent_a.id in merged_ids or agent_b.id in merged_ids:
                continue

            # Create and execute merge
            merged = self.merge_agents(agent_a, agent_b, current_iteration)
            self.execute_merge(agent_a, agent_b, merged, current_iteration)

            merged_pairs.append((agent_a.id, agent_b.id))
            merged_ids.add(agent_a.id)
            merged_ids.add(agent_b.id)

        return {
            'candidates': len(candidates),
            'merged': len(merged_pairs),
            'merged_pairs': merged_pairs
        }

    def update_threshold(self, new_threshold: float):
        """Update similarity threshold."""
        self.similarity_threshold = max(0.5, min(0.98, new_threshold))

    def get_statistics(self) -> dict:
        """Get merging statistics."""
        return {
            'total_merges': self.total_merges,
            'coactivation_pairs_tracked': len(self.coactivation_counts),
            'total_activations_tracked': self.total_activations
        }
