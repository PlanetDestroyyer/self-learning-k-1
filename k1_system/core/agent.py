"""
Agent class for the Self-Learning K-1 System.

Each agent represents a node in the hierarchical neural network,
containing neural network weights, trust scores, and structural information.
"""

import numpy as np
from typing import Optional, List, Set
from collections import deque


class Agent:
    """
    Represents a single agent (node) in the hierarchical knowledge system.

    Attributes:
        id: Unique identifier for the agent
        agent_type: Type of agent ('master', 'manager', 'agent', 'sub_agent')
        specialty: Domain or specialty this agent handles
        weights: Neural network weights
        trust: Trust score (0.0 to 1.0)
        parent: Reference to parent agent
        children: Set of child agents
        creation_iteration: When this agent was created
        last_used: Last iteration this agent was activated
        usage_count: Total number of times activated
        activation_history: Recent activation levels
        error_history: Recent errors this agent contributed to
    """

    _id_counter = 0

    def __init__(self,
                 agent_id: Optional[str] = None,
                 agent_type: str = 'agent',
                 specialty: str = 'general',
                 input_dim: int = 128,
                 hidden_dim: int = 64,
                 output_dim: int = 128,
                 initial_trust: float = 0.3,
                 creation_iteration: int = 0):
        """
        Initialize an agent.

        Args:
            agent_id: Unique ID (auto-generated if None)
            agent_type: Type of agent
            specialty: Domain specialty
            input_dim: Input dimension for neural network
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            initial_trust: Starting trust score
            creation_iteration: Iteration when created
        """
        if agent_id is None:
            agent_id = f"{agent_type}_{Agent._id_counter}"
            Agent._id_counter += 1

        self.id = agent_id
        self.agent_type = agent_type
        self.specialty = specialty

        # Neural network weights (simple 2-layer network)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights with Xavier initialization
        self.weights = {
            'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros(output_dim),
            # Routing weights (for deciding which child to activate)
            'routing': np.random.randn(output_dim, 10) * 0.01  # 10 potential children max initially
        }

        # Trust and performance tracking
        self.trust = initial_trust
        self.creation_iteration = creation_iteration
        self.last_used = creation_iteration
        self.usage_count = 0
        self.usage_count_last_10k = 0
        self.last_usage_reset = creation_iteration

        # Activation and error tracking
        self.activation_history = deque(maxlen=1000)  # Last 1000 activations
        self.error_history = deque(maxlen=1000)  # Last 1000 errors
        self.activation_level = 0.0  # Current activation level

        # Hierarchical structure
        self.parent: Optional[Agent] = None
        self.children: Set[Agent] = set()

        # Performance metrics
        self.success_count = 0
        self.failure_count = 0
        self.total_error_reduction = 0.0
        self.protected = False  # Protection from deletion
        self.protected_until = 0  # Iteration until which agent is protected

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through this agent's neural network.

        Args:
            x: Input vector

        Returns:
            Output vector
        """
        # Layer 1
        h = np.maximum(0, x @ self.weights['W1'] + self.weights['b1'])  # ReLU activation

        # Layer 2
        output = h @ self.weights['W2'] + self.weights['b2']

        return output

    def route(self, x: np.ndarray) -> np.ndarray:
        """
        Compute routing scores for children.

        Args:
            x: Input vector (typically output from forward pass)

        Returns:
            Routing scores for each child
        """
        if len(self.children) == 0:
            return np.array([])

        # Compute routing scores
        routing_scores = x @ self.weights['routing'][:, :len(self.children)]

        # Apply softmax to get probabilities
        exp_scores = np.exp(routing_scores - np.max(routing_scores))
        routing_probs = exp_scores / np.sum(exp_scores)

        return routing_probs

    def update_weights(self, gradient: dict, learning_rate: float):
        """
        Update agent's weights using gradients.

        Args:
            gradient: Dictionary of gradients for each weight
            learning_rate: Learning rate for update
        """
        for key in self.weights:
            if key in gradient:
                self.weights[key] -= learning_rate * gradient[key]

    def record_activation(self, activation_level: float, iteration: int):
        """
        Record activation for this iteration.

        Args:
            activation_level: Activation level (0.0 to 1.0)
            iteration: Current iteration number
        """
        self.activation_level = activation_level
        self.activation_history.append(activation_level)
        self.last_used = iteration
        self.usage_count += 1
        self.usage_count_last_10k += 1

    def record_error(self, error_magnitude: float):
        """
        Record error contribution.

        Args:
            error_magnitude: Magnitude of error
        """
        self.error_history.append(error_magnitude)

    def reset_usage_counter(self, iteration: int):
        """Reset the recent usage counter (called every 10k iterations)."""
        self.usage_count_last_10k = 0
        self.last_usage_reset = iteration

    def add_child(self, child: 'Agent'):
        """Add a child agent."""
        self.children.add(child)
        child.parent = self

        # Expand routing weights if needed
        if len(self.children) > self.weights['routing'].shape[1]:
            new_routing = np.random.randn(self.output_dim, len(self.children)) * 0.01
            new_routing[:, :self.weights['routing'].shape[1]] = self.weights['routing']
            self.weights['routing'] = new_routing

    def remove_child(self, child: 'Agent'):
        """Remove a child agent."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def is_only_agent_in_domain(self) -> bool:
        """Check if this is the only agent under its parent."""
        if self.parent is None:
            return True
        return len(self.parent.children) <= 1

    def trust_increasing(self, window: int = 100) -> bool:
        """
        Check if trust is increasing over recent history.

        Args:
            window: Number of recent iterations to check

        Returns:
            True if trust is increasing
        """
        if len(self.activation_history) < window:
            return True  # Not enough history, give benefit of doubt

        recent = list(self.activation_history)[-window:]
        first_half = np.mean(recent[:window//2])
        second_half = np.mean(recent[window//2:])

        return second_half > first_half

    def mark_protected(self, until_iteration: int):
        """Mark agent as protected from deletion until specified iteration."""
        self.protected = True
        self.protected_until = until_iteration

    def check_protection(self, current_iteration: int):
        """Check and update protection status."""
        if self.protected and current_iteration >= self.protected_until:
            self.protected = False

    def compute_responsibility(self, error_magnitude: float) -> float:
        """
        Compute responsibility for an error.

        Args:
            error_magnitude: Magnitude of the error

        Returns:
            Responsibility score
        """
        return self.activation_level * error_magnitude

    def compute_ranking_score(self, responsibility: float) -> float:
        """
        Compute ranking score for credit assignment.

        Args:
            responsibility: Responsibility score

        Returns:
            Ranking score (higher = more likely to be updated)
        """
        return responsibility * (1.0 + self.trust)

    def get_avg_activation(self, window: int = 1000) -> float:
        """Get average activation over recent window."""
        if len(self.activation_history) == 0:
            return 0.0
        recent = list(self.activation_history)[-window:]
        return np.mean(recent)

    def get_avg_error(self, window: int = 1000) -> float:
        """Get average error over recent window."""
        if len(self.error_history) == 0:
            return 0.0
        recent = list(self.error_history)[-window:]
        return np.mean(recent)

    def __repr__(self) -> str:
        return (f"Agent(id={self.id}, type={self.agent_type}, specialty={self.specialty}, "
                f"trust={self.trust:.3f}, children={len(self.children)})")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return self.id == other.id
