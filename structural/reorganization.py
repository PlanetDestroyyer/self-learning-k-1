"""
Self-reorganization system for K-1 Self-Learning System.

Optimizes hierarchy structure based on usage patterns.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from core.agent import Agent
from core.hierarchy import Hierarchy


class ReorganizationSystem:
    """
    Manages autonomous hierarchy reorganization.

    Reorganization includes:
    - Moving misplaced agents
    - Splitting overloaded managers
    - Consolidating underused managers
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        max_moves_per_cycle: float = 0.1,
        min_children_to_split: int = 10,
        max_children_per_manager: int = 20
    ):
        """
        Initialize reorganization system.

        Args:
            hierarchy: Hierarchy to reorganize
            max_moves_per_cycle: Max fraction of agents to move
            min_children_to_split: Min children to consider splitting manager
            max_children_per_manager: Trigger split above this
        """
        self.hierarchy = hierarchy
        self.max_moves_per_cycle = max_moves_per_cycle
        self.min_children_to_split = min_children_to_split
        self.max_children_per_manager = max_children_per_manager

        # Coactivation tracking
        self.coactivation_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_activations = 0

        self.reorganization_history = []
        self.total_moves = 0

    def record_activation_pattern(self, activated_agents: List[Agent]):
        """
        Record which agents activated together.

        Args:
            activated_agents: Agents activated in this forward pass
        """
        self.total_activations += 1

        # Record coactivations
        for i, agent_a in enumerate(activated_agents):
            for agent_b in activated_agents[i+1:]:
                self.coactivation_matrix[agent_a.id][agent_b.id] += 1
                self.coactivation_matrix[agent_b.id][agent_a.id] += 1

    def find_optimal_parent(self, agent: Agent) -> Agent:
        """
        Find optimal parent for an agent based on coactivation.

        Args:
            agent: Agent to find parent for

        Returns:
            Optimal parent agent
        """
        coactivations = self.coactivation_matrix.get(agent.id, {})

        if not coactivations:
            return agent.parent

        # Score each potential parent by coactivation with their children
        parent_scores = defaultdict(int)

        for other_id, count in coactivations.items():
            other = self.hierarchy.find_agent(other_id)
            if other and other.parent:
                parent_scores[other.parent.id] += count

        if not parent_scores:
            return agent.parent

        # Find best parent
        best_parent_id = max(parent_scores, key=parent_scores.get)
        best_parent = self.hierarchy.find_agent(best_parent_id)

        return best_parent if best_parent else agent.parent

    def find_misplaced_agents(self) -> List[Tuple[Agent, Agent]]:
        """
        Find agents that would be better under different parents.

        Returns:
            List of (agent, optimal_parent) tuples
        """
        misplaced = []

        for agent in self.hierarchy.get_all_agents():
            if agent == self.hierarchy.root:
                continue

            optimal = self.find_optimal_parent(agent)

            if optimal and optimal != agent.parent:
                misplaced.append((agent, optimal))

        return misplaced

    def move_agent(self, agent: Agent, new_parent: Agent) -> bool:
        """
        Move an agent to a new parent.

        Args:
            agent: Agent to move
            new_parent: New parent

        Returns:
            True if successful
        """
        if self.hierarchy.move_agent(agent, new_parent):
            self.total_moves += 1
            return True
        return False

    def split_manager(self, manager: Agent, current_iteration: int) -> Tuple[Agent, Agent]:
        """
        Split a manager with too many children.

        Args:
            manager: Manager to split
            current_iteration: Current iteration

        Returns:
            Tuple of new managers
        """
        children = list(manager.children)

        if len(children) < self.min_children_to_split:
            return None, None

        # Split by trust (high trust vs low trust)
        children_sorted = sorted(children, key=lambda a: a.trust, reverse=True)
        mid = len(children_sorted) // 2

        group_high = children_sorted[:mid]
        group_low = children_sorted[mid:]

        # Create two new managers
        manager1 = Agent(
            agent_id=f"{manager.id}_split_high_{current_iteration}",
            agent_type='manager',
            specialty=f"{manager.specialty}_HighTrust",
            input_dim=manager.input_dim,
            hidden_dim=manager.hidden_dim,
            output_dim=manager.output_dim,
            initial_trust=manager.trust,
            creation_iteration=current_iteration
        )

        manager2 = Agent(
            agent_id=f"{manager.id}_split_low_{current_iteration}",
            agent_type='manager',
            specialty=f"{manager.specialty}_LowTrust",
            input_dim=manager.input_dim,
            hidden_dim=manager.hidden_dim,
            output_dim=manager.output_dim,
            initial_trust=manager.trust,
            creation_iteration=current_iteration
        )

        # Copy weights
        for key in manager.weights:
            manager1.weights[key] = manager.weights[key].copy()
            manager2.weights[key] = manager.weights[key].copy()

        # Reassign children
        parent = manager.parent

        if parent:
            self.hierarchy.add_agent(manager1, parent)
            self.hierarchy.add_agent(manager2, parent)

        for child in group_high:
            manager.remove_child(child)
            manager1.add_child(child)

        for child in group_low:
            manager.remove_child(child)
            manager2.add_child(child)

        # Remove old manager
        self.hierarchy.remove_agent(manager)

        return manager1, manager2

    def consolidate_manager(self, manager: Agent) -> bool:
        """
        Consolidate a rarely-used manager by moving children to parent.

        Args:
            manager: Manager to consolidate

        Returns:
            True if consolidated
        """
        if manager.parent is None:
            return False

        if manager == self.hierarchy.root:
            return False

        parent = manager.parent

        # Move all children to grandparent
        for child in list(manager.children):
            manager.remove_child(child)
            parent.add_child(child)

        # Remove manager
        self.hierarchy.remove_agent(manager)

        return True

    def execute_reorganization(self, current_iteration: int) -> dict:
        """
        Execute reorganization cycle.

        Args:
            current_iteration: Current iteration

        Returns:
            Statistics
        """
        results = {
            'agents_moved': 0,
            'managers_split': 0,
            'managers_consolidated': 0
        }

        # Find and move misplaced agents
        misplaced = self.find_misplaced_agents()
        max_moves = max(1, int(self.hierarchy.count_agents() * self.max_moves_per_cycle))

        for agent, optimal_parent in misplaced[:max_moves]:
            if self.move_agent(agent, optimal_parent):
                results['agents_moved'] += 1

        # Check for managers to split
        for manager in self.hierarchy.get_agents_by_type('manager'):
            if len(manager.children) > self.max_children_per_manager:
                m1, m2 = self.split_manager(manager, current_iteration)
                if m1 and m2:
                    results['managers_split'] += 1

        # Check for managers to consolidate
        for manager in list(self.hierarchy.get_agents_by_type('manager')):
            if manager.get_avg_activation() < 0.05 and len(manager.children) < 3:
                if self.consolidate_manager(manager):
                    results['managers_consolidated'] += 1

        # Record
        self.reorganization_history.append({
            'iteration': current_iteration,
            **results
        })

        return results

    def get_statistics(self) -> dict:
        """Get reorganization statistics."""
        return {
            'total_moves': self.total_moves,
            'coactivation_pairs': sum(len(v) for v in self.coactivation_matrix.values()) // 2,
            'total_activations_tracked': self.total_activations
        }
