"""
Routing system for hierarchical forward pass.

Implements multi-level routing from Master Manager down through
the hierarchy to find the most appropriate specialist agents.
"""

from typing import List, Tuple, Optional
import numpy as np
from .agent import Agent
from .hierarchy import Hierarchy


class RoutingPath:
    """
    Represents a path through the hierarchy during routing.

    Attributes:
        agents: List of agents in the path (root to leaf)
        confidences: Confidence at each routing decision
        activations: Activation levels for each agent
        final_output: Final output from the path
    """

    def __init__(self):
        self.agents: List[Agent] = []
        self.confidences: List[float] = []
        self.activations: List[float] = []
        self.final_output: Optional[np.ndarray] = None

    def add_step(self, agent: Agent, confidence: float, activation: float):
        """Add a routing step."""
        self.agents.append(agent)
        self.confidences.append(confidence)
        self.activations.append(activation)

    def get_activated_agents(self) -> List[Agent]:
        """Get list of activated agents."""
        return self.agents

    def get_avg_confidence(self) -> float:
        """Get average confidence across path."""
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))

    def get_path_length(self) -> int:
        """Get length of routing path."""
        return len(self.agents)


class HierarchicalRouter:
    """
    Manages hierarchical routing through the agent tree.
    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 confidence_threshold: float = 0.5,
                 max_depth: int = 6,
                 exploration_rate: float = 0.1):
        """
        Initialize router.

        Args:
            hierarchy: Hierarchy to route through
            confidence_threshold: Minimum confidence to continue routing
            max_depth: Maximum routing depth
            exploration_rate: Probability of exploring sub-optimal paths
        """
        self.hierarchy = hierarchy
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth
        self.exploration_rate = exploration_rate

    def route(self, x: np.ndarray, mode: str = 'hard') -> RoutingPath:
        """
        Route input through hierarchy.

        Args:
            x: Input vector
            mode: Routing mode ('hard' for single path, 'soft' for multiple paths)

        Returns:
            RoutingPath object with activated agents
        """
        if mode == 'hard':
            return self._hard_route(x)
        elif mode == 'soft':
            return self._soft_route(x)
        else:
            raise ValueError(f"Unknown routing mode: {mode}")

    def _hard_route(self, x: np.ndarray) -> RoutingPath:
        """
        Hard routing: Single path activation.

        Args:
            x: Input vector

        Returns:
            Routing path
        """
        path = RoutingPath()

        # Start at root
        current_agent = self.hierarchy.root
        current_input = x
        depth = 0

        while current_agent is not None and depth < self.max_depth:
            # Forward pass through current agent
            output = current_agent.forward(current_input)

            # Record activation (full activation for hard routing)
            activation = 1.0
            current_agent.record_activation(activation, 0)  # Iteration will be updated later

            # Check if we should continue routing
            if len(current_agent.child_agents) == 0:
                # Leaf node, stop
                confidence = 1.0
                path.add_step(current_agent, confidence, activation)
                path.final_output = output
                break

            # Compute routing scores for children
            routing_scores = current_agent.route(output)

            # Convert to numpy if it's a tensor
            if hasattr(routing_scores, 'detach'):
                routing_scores_np = routing_scores.detach().cpu().numpy()
            else:
                routing_scores_np = routing_scores

            # Check if confident enough to route
            max_score = np.max(routing_scores_np)
            confidence = max_score

            if confidence < self.confidence_threshold:
                # Not confident enough, stop here
                path.add_step(current_agent, confidence, activation)
                path.final_output = output
                break

            # Select next agent (with exploration)
            if np.random.random() < self.exploration_rate:
                # Explore: random child
                next_idx = np.random.randint(len(routing_scores_np))
            else:
                # Exploit: best child
                next_idx = np.argmax(routing_scores_np)

            # Add current step to path
            path.add_step(current_agent, confidence, activation)

            # Move to next agent
            children_list = list(current_agent.child_agents)
            current_agent = children_list[next_idx]
            current_input = output
            depth += 1

        return path

    def _soft_route(self, x: np.ndarray, top_k: int = 3) -> RoutingPath:
        """
        Soft routing: Multiple paths with weights.

        Routes to top-k children at each level based on routing scores.

        Args:
            x: Input vector
            top_k: Number of paths to explore

        Returns:
            Combined routing path
        """
        path = RoutingPath()

        # Start at root
        current_level = [(self.hierarchy.root, x, 1.0)]  # (agent, input, weight)
        depth = 0

        all_activated = []

        while current_level and depth < self.max_depth:
            next_level = []

            for agent, input_vec, weight in current_level:
                # Forward pass
                output = agent.forward(input_vec)

                # Record activation weighted by path probability
                agent.record_activation(weight, 0)
                all_activated.append((agent, weight, depth))

                # Check for children
                if len(agent.child_agents) == 0:
                    # Leaf node
                    path.add_step(agent, 1.0, weight)
                    if path.final_output is None:
                        path.final_output = output * weight
                    else:
                        path.final_output += output * weight
                    continue

                # Compute routing scores
                routing_scores = agent.route(output)
                
                # Convert to numpy if tensor
                if hasattr(routing_scores, 'detach'):
                    routing_scores_np = routing_scores.detach().cpu().numpy()
                else:
                    routing_scores_np = np.array(routing_scores)

                # Get top-k children
                if len(routing_scores_np) <= top_k:
                    top_indices = np.argsort(routing_scores_np)[::-1]
                else:
                    top_indices = np.argsort(routing_scores_np)[::-1][:top_k]

                # Add top-k to next level
                for idx in top_indices:
                    child = list(agent.child_agents)[idx]
                    child_weight = weight * float(routing_scores_np[idx])

                    if child_weight > 0.01:  # Only keep significant paths
                        next_level.append((child, output, child_weight))

            # Normalize weights at this level
            if next_level:
                total_weight = sum(w for _, _, w in next_level)
                next_level = [(a, i, w/total_weight) for a, i, w in next_level]

            current_level = next_level
            depth += 1

        # Add all activated agents to path
        all_activated.sort(key=lambda x: x[2])  # Sort by depth
        for agent, weight, depth_val in all_activated:
            path.add_step(agent, weight, weight)

        return path

    def route_with_cache(self,
                        x: np.ndarray,
                        trust_cache: 'TrustCache',
                        error_type: Optional[str] = None) -> RoutingPath:
        """
        Route with trust cache lookup.

        First checks cache for specialists, then falls back to hierarchical routing.

        Args:
            x: Input vector
            trust_cache: Trust cache to check
            error_type: Type of error (if known)

        Returns:
            Routing path
        """
        # Try cache first
        if error_type and trust_cache.get_cache_size() > 0:
            specialists = trust_cache.get_specialists_for_error_type(error_type)

            if specialists:
                # Use top specialist from cache
                path = RoutingPath()
                specialist = specialists[0]

                # Run through specialist
                output = specialist.forward(x)
                specialist.record_activation(1.0, 0)

                path.add_step(specialist, 1.0, 1.0)
                path.final_output = output

                return path

        # Fall back to hierarchical routing
        return self.route(x, mode='hard')

    def get_routing_statistics(self, paths: List[RoutingPath]) -> dict:
        """
        Get statistics about routing behavior.

        Args:
            paths: List of routing paths from recent iterations

        Returns:
            Dictionary of statistics
        """
        if not paths:
            return {}

        path_lengths = [p.get_path_length() for p in paths]
        avg_confidences = [p.get_avg_confidence() for p in paths]

        return {
            'avg_path_length': float(np.mean(path_lengths)),
            'std_path_length': float(np.std(path_lengths)),
            'min_path_length': int(np.min(path_lengths)),
            'max_path_length': int(np.max(path_lengths)),
            'avg_confidence': float(np.mean(avg_confidences)),
            'std_confidence': float(np.std(avg_confidences))
        }

    def update_exploration_rate(self, new_rate: float):
        """
        Update exploration rate.

        Args:
            new_rate: New exploration rate
        """
        self.exploration_rate = max(0.0, min(1.0, new_rate))

    def compute_exploration_rate(self,
                                iteration: int,
                                initial_rate: float = 0.3,
                                decay_constant: float = 100000,
                                minimum_rate: float = 0.05) -> float:
        """
        Compute exploration rate with exponential decay.

        Args:
            iteration: Current iteration
            initial_rate: Initial exploration rate
            decay_constant: Decay constant
            minimum_rate: Minimum exploration rate

        Returns:
            Exploration rate
        """
        rate = initial_rate * np.exp(-iteration / decay_constant)
        return max(minimum_rate, rate)
