"""
Agent class for K-1 Self-Learning System.

Each agent is a node in the hierarchical network with its own neural network weights,
trust score, and structural information.
"""

import numpy as np
from typing import Optional, Set, Dict
from collections import deque


class Agent:
    """
    A single agent (node) in the hierarchical knowledge system.

    Each agent contains:
    - Neural network weights (2-layer MLP)
    - Trust score (0.0 to 1.0)
    - Hierarchical connections (parent/children)
    - Performance tracking
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str = 'agent',
        specialty: str = 'general',
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        initial_trust: float = 0.3,
        creation_iteration: int = 0
    ):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier
            agent_type: 'master', 'manager', 'agent', or 'sub_agent'
            specialty: Domain specialty description
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            initial_trust: Starting trust score
            creation_iteration: When this agent was created
        """
        self.id = agent_id
        self.agent_type = agent_type
        self.specialty = specialty

        # Dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Neural network weights (Xavier initialization)
        self.weights = {
            'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros(output_dim),
            'routing': np.random.randn(output_dim, 20) * 0.01  # For child selection
        }

        # Trust and performance
        self.trust = initial_trust
        self.creation_iteration = creation_iteration
        self.last_used = creation_iteration
        self.usage_count = 0
        self.usage_count_window = 0  # Recent usage count
        self.window_start = creation_iteration

        # Activation tracking
        self.activation_level = 0.0
        self.activation_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
        self.last_hidden = None  # Cache for backprop

        # Hierarchy
        self.parent: Optional['Agent'] = None
        self.children: Set['Agent'] = set()
        self._children_order: list = []  # Maintain consistent ordering

        # Performance metrics
        self.success_count = 0
        self.failure_count = 0
        self.total_error_reduction = 0.0

        # Protection from deletion
        self.protected = False
        self.protected_until = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through agent's neural network.

        Args:
            x: Input vector (input_dim,)

        Returns:
            Output vector (output_dim,)
        """
        # Layer 1 with ReLU
        self.last_hidden = np.maximum(0, x @ self.weights['W1'] + self.weights['b1'])

        # Layer 2 (linear output)
        output = self.last_hidden @ self.weights['W2'] + self.weights['b2']

        return output

    def get_routing_scores(self) -> np.ndarray:
        """
        Get routing scores for children based on last forward pass.

        Returns:
            Softmax probabilities for each child
        """
        if len(self._children_order) == 0:
            return np.array([])

        n_children = len(self._children_order)
        scores = self.last_hidden @ self.weights['routing'][:, :n_children]

        # Softmax
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(np.clip(scores_shifted, -20, 20))
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)

        return probs

    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float):
        """
        Update weights using computed gradients.

        Args:
            gradients: Dictionary of gradients for each weight
            learning_rate: Learning rate
        """
        for key in ['W1', 'b1', 'W2', 'b2']:
            if key in gradients and gradients[key] is not None:
                self.weights[key] -= learning_rate * gradients[key]

    def record_activation(self, activation_level: float, iteration: int):
        """Record activation for this iteration."""
        self.activation_level = activation_level
        self.activation_history.append(activation_level)
        self.last_used = iteration
        self.usage_count += 1
        self.usage_count_window += 1

    def record_error(self, error_magnitude: float):
        """Record error contribution."""
        self.error_history.append(error_magnitude)

    def reset_window_counter(self, iteration: int):
        """Reset the window-based usage counter."""
        self.usage_count_window = 0
        self.window_start = iteration

    def add_child(self, child: 'Agent'):
        """Add a child agent."""
        self.children.add(child)
        child.parent = self
        self._update_children_order()

        # Expand routing weights if needed
        if len(self._children_order) > self.weights['routing'].shape[1]:
            new_routing = np.random.randn(self.output_dim, len(self._children_order) + 10) * 0.01
            new_routing[:, :self.weights['routing'].shape[1]] = self.weights['routing']
            self.weights['routing'] = new_routing

    def remove_child(self, child: 'Agent'):
        """Remove a child agent."""
        if child in self.children:
            self.children.discard(child)
            child.parent = None
            self._update_children_order()

    def _update_children_order(self):
        """Maintain consistent ordering of children by ID."""
        self._children_order = sorted(list(self.children), key=lambda a: a.id)

    def get_children_ordered(self) -> list:
        """Get children in consistent order."""
        return self._children_order

    def is_only_child(self) -> bool:
        """Check if this is the only child of its parent."""
        if self.parent is None:
            return True
        return len(self.parent.children) <= 1

    def is_trust_increasing(self, window: int = 100) -> bool:
        """Check if trust trend is positive."""
        if len(self.activation_history) < window:
            return True  # Benefit of doubt

        recent = list(self.activation_history)[-window:]
        first_half = np.mean(recent[:window//2])
        second_half = np.mean(recent[window//2:])

        return second_half >= first_half

    def mark_protected(self, until_iteration: int):
        """Mark agent as protected from deletion."""
        self.protected = True
        self.protected_until = until_iteration

    def check_protection(self, current_iteration: int):
        """Update protection status."""
        if self.protected and current_iteration >= self.protected_until:
            self.protected = False

    def get_avg_activation(self, window: int = 1000) -> float:
        """Get average activation over recent window."""
        if len(self.activation_history) == 0:
            return 0.0
        recent = list(self.activation_history)[-window:]
        return float(np.mean(recent))

    def get_avg_error(self, window: int = 1000) -> float:
        """Get average error over recent window."""
        if len(self.error_history) == 0:
            return 0.0
        recent = list(self.error_history)[-window:]
        return float(np.mean(recent))

    def compute_responsibility(self, error_magnitude: float) -> float:
        """Compute responsibility for an error."""
        return self.activation_level * error_magnitude

    def compute_ranking_score(self, responsibility: float) -> float:
        """Compute ranking score for credit assignment."""
        return responsibility * (1.0 + self.trust)

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, type={self.agent_type}, trust={self.trust:.3f})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return self.id == other.id
