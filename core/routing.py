"""
Hierarchical routing for K-1 Self-Learning System.

Routes inputs through the hierarchy to find specialist agents.
"""

import numpy as np
from typing import List, Optional, Tuple
from .agent import Agent
from .hierarchy import Hierarchy


class RoutingPath:
    """Records the path taken through the hierarchy."""

    def __init__(self):
        self.path: List[Agent] = []
        self.activation_levels: List[float] = []
        self.routing_decisions: List[Tuple[Agent, int, float]] = []  # (agent, child_idx, confidence)

    def add_step(self, agent: Agent, activation_level: float, child_idx: int = -1, confidence: float = 1.0):
        """Add a routing step."""
        self.path.append(agent)
        self.activation_levels.append(activation_level)
        if child_idx >= 0:
            self.routing_decisions.append((agent, child_idx, confidence))

    def get_activated_agents(self) -> List[Agent]:
        """Get all agents that were activated."""
        return self.path

    def get_leaf_agent(self) -> Optional[Agent]:
        """Get the final agent in the path."""
        return self.path[-1] if self.path else None

    def get_total_activation(self) -> float:
        """Get sum of activation levels."""
        return sum(self.activation_levels)


class HierarchicalRouter:
    """
    Routes inputs through the agent hierarchy.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        confidence_threshold: float = 0.5,
        max_depth: int = 6,
        exploration_rate: float = 0.1
    ):
        """
        Initialize router.

        Args:
            hierarchy: The hierarchy to route through
            confidence_threshold: Minimum confidence to continue routing
            max_depth: Maximum routing depth
            exploration_rate: Probability of random exploration
        """
        self.hierarchy = hierarchy
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth
        self.exploration_rate = exploration_rate
        self.current_iteration = 0

    def update_iteration(self, iteration: int):
        """Update current iteration (for exploration decay)."""
        self.current_iteration = iteration

    def get_exploration_rate(self) -> float:
        """Get current exploration rate with decay."""
        decay = np.exp(-self.current_iteration / 100000)
        return max(0.05, self.exploration_rate * decay)

    def route(self, x: np.ndarray, mode: str = 'hard') -> Tuple[np.ndarray, RoutingPath]:
        """
        Route input through hierarchy.

        Args:
            x: Input vector
            mode: 'hard' (single path) or 'soft' (weighted combination)

        Returns:
            (output, routing_path)
        """
        if mode == 'hard':
            return self._route_hard(x)
        else:
            return self._route_soft(x)

    def _route_hard(self, x: np.ndarray) -> Tuple[np.ndarray, RoutingPath]:
        """Route through single path (greedy routing)."""
        path = RoutingPath()

        if self.hierarchy.root is None:
            return x, path

        current_agent = self.hierarchy.root
        current_input = x
        depth = 0

        while current_agent is not None and depth < self.max_depth:
            # Forward through current agent
            output = current_agent.forward(current_input)
            activation_level = float(np.mean(np.abs(output)))
            current_agent.record_activation(activation_level, self.current_iteration)

            # Check for children
            children = current_agent.get_children_ordered()

            if len(children) == 0:
                # Leaf node - done
                path.add_step(current_agent, activation_level)
                return output, path

            # Get routing scores
            routing_scores = current_agent.get_routing_scores()

            if len(routing_scores) == 0:
                path.add_step(current_agent, activation_level)
                return output, path

            # Select next agent
            exploration_rate = self.get_exploration_rate()

            if np.random.random() < exploration_rate:
                # Random exploration
                next_idx = np.random.randint(len(children))
                confidence = routing_scores[next_idx]
            else:
                # Greedy selection
                next_idx = np.argmax(routing_scores)
                confidence = float(routing_scores[next_idx])

            path.add_step(current_agent, activation_level, next_idx, confidence)

            # Stop if confidence too low
            if confidence < self.confidence_threshold and depth > 0:
                return output, path

            # Move to next agent
            current_agent = children[next_idx]
            current_input = output
            depth += 1

        return output, path

    def _route_soft(self, x: np.ndarray) -> Tuple[np.ndarray, RoutingPath]:
        """Route through weighted combination of paths."""
        path = RoutingPath()

        if self.hierarchy.root is None:
            return x, path

        # For soft routing, we do weighted sum of all agents at each level
        outputs = []
        weights = []

        def traverse(agent: Agent, input_x: np.ndarray, weight: float, depth: int):
            if depth >= self.max_depth:
                return

            # Forward through agent
            output = agent.forward(input_x)
            activation_level = float(np.mean(np.abs(output)))
            agent.record_activation(activation_level * weight, self.current_iteration)

            children = agent.get_children_ordered()

            if len(children) == 0:
                # Leaf - record output
                outputs.append(output)
                weights.append(weight)
                path.add_step(agent, activation_level * weight)
                return

            # Route to children with weights
            routing_scores = agent.get_routing_scores()

            for idx, child in enumerate(children):
                if idx < len(routing_scores):
                    child_weight = weight * routing_scores[idx]
                    if child_weight > 0.01:  # Prune very low weights
                        traverse(child, output, child_weight, depth + 1)

        traverse(self.hierarchy.root, x, 1.0, 0)

        if len(outputs) == 0:
            return x, path

        # Weighted average of outputs
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)

        final_output = sum(w * o for w, o in zip(weights, outputs))

        return final_output, path

    def update_exploration_rate(self, new_rate: float):
        """Update base exploration rate."""
        self.exploration_rate = max(0.0, min(1.0, new_rate))

    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
