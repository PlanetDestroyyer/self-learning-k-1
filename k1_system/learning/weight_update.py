"""
Weight update system for the Self-Learning K-1 System.

Implements gradient computation and weight updates for selected agents.
"""

import numpy as np
from typing import List, Dict
from ..core.agent import Agent


class WeightUpdater:
    """
    Manages weight updates for agents using computed gradients.
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize weight updater.

        Args:
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.update_count = 0

    def compute_gradient(self,
                        agent: Agent,
                        x: np.ndarray,
                        target: np.ndarray,
                        output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradient for an agent's weights.

        Uses simple backpropagation through the agent's local network.

        Args:
            agent: Agent to compute gradient for
            x: Input to agent
            target: Target output
            output: Actual output from agent

        Returns:
            Dictionary of gradients for each weight
        """
        gradients = {}

        # Forward pass to get intermediate activations
        h = np.maximum(0, x @ agent.weights['W1'] + agent.weights['b1'])  # ReLU
        pred = h @ agent.weights['W2'] + agent.weights['b2']

        # Output layer gradient
        d_output = pred - target  # Assuming MSE loss

        # Gradient for W2 and b2
        gradients['W2'] = np.outer(h, d_output)
        gradients['b2'] = d_output

        # Backpropagate to hidden layer
        d_h = d_output @ agent.weights['W2'].T
        d_h[h <= 0] = 0  # ReLU gradient

        # Gradient for W1 and b1
        gradients['W1'] = np.outer(x, d_h)
        gradients['b1'] = d_h

        return gradients

    def update_agent(self,
                    agent: Agent,
                    gradient: Dict[str, np.ndarray]):
        """
        Update an agent's weights using gradient.

        Args:
            agent: Agent to update
            gradient: Gradient dictionary
        """
        agent.update_weights(gradient, self.learning_rate)
        self.update_count += 1

    def update_agents(self,
                     agents: List[Agent],
                     gradients: List[Dict[str, np.ndarray]]):
        """
        Update multiple agents with their gradients.

        Args:
            agents: List of agents to update
            gradients: List of gradient dictionaries
        """
        for agent, gradient in zip(agents, gradients):
            self.update_agent(agent, gradient)

    def update_learning_rate(self, new_rate: float):
        """
        Update learning rate.

        Args:
            new_rate: New learning rate
        """
        self.learning_rate = max(0.0, new_rate)

    def compute_gradient_norm(self, gradient: Dict[str, np.ndarray]) -> float:
        """
        Compute L2 norm of gradient.

        Args:
            gradient: Gradient dictionary

        Returns:
            Gradient norm
        """
        total_norm = 0.0
        for key, grad in gradient.items():
            total_norm += np.sum(grad ** 2)
        return np.sqrt(total_norm)

    def clip_gradient(self,
                     gradient: Dict[str, np.ndarray],
                     max_norm: float = 5.0) -> Dict[str, np.ndarray]:
        """
        Clip gradient by norm to prevent exploding gradients.

        Args:
            gradient: Gradient dictionary
            max_norm: Maximum allowed norm

        Returns:
            Clipped gradient
        """
        grad_norm = self.compute_gradient_norm(gradient)

        if grad_norm > max_norm:
            # Scale down gradient
            scale = max_norm / grad_norm
            clipped = {key: grad * scale for key, grad in gradient.items()}
            return clipped
        else:
            return gradient

    def get_update_statistics(self) -> Dict:
        """
        Get statistics about weight updates.

        Returns:
            Dictionary of statistics
        """
        return {
            'total_updates': self.update_count,
            'current_learning_rate': self.learning_rate
        }


class AdaptiveWeightUpdater(WeightUpdater):
    """
    Weight updater with adaptive learning rate (Adam-like).
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize adaptive weight updater.

        Args:
            learning_rate: Base learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Momentum terms for each agent
        self.m: Dict[str, Dict[str, np.ndarray]] = {}  # First moment
        self.v: Dict[str, Dict[str, np.ndarray]] = {}  # Second moment
        self.t: Dict[str, int] = {}  # Time step for each agent

    def update_agent(self,
                    agent: Agent,
                    gradient: Dict[str, np.ndarray]):
        """
        Update agent weights using adaptive learning rate.

        Args:
            agent: Agent to update
            gradient: Gradient dictionary
        """
        agent_id = agent.id

        # Initialize momentum terms if needed
        if agent_id not in self.m:
            self.m[agent_id] = {key: np.zeros_like(grad) for key, grad in gradient.items()}
            self.v[agent_id] = {key: np.zeros_like(grad) for key, grad in gradient.items()}
            self.t[agent_id] = 0

        # Increment time step
        self.t[agent_id] += 1
        t = self.t[agent_id]

        # Update biased first and second moments
        for key in gradient:
            self.m[agent_id][key] = self.beta1 * self.m[agent_id][key] + (1 - self.beta1) * gradient[key]
            self.v[agent_id][key] = self.beta2 * self.v[agent_id][key] + (1 - self.beta2) * (gradient[key] ** 2)

            # Bias correction
            m_hat = self.m[agent_id][key] / (1 - self.beta1 ** t)
            v_hat = self.v[agent_id][key] / (1 - self.beta2 ** t)

            # Update weights
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            agent.weights[key] -= update

        self.update_count += 1

    def reset_agent_state(self, agent_id: str):
        """
        Reset momentum state for an agent.

        Args:
            agent_id: Agent ID
        """
        if agent_id in self.m:
            del self.m[agent_id]
        if agent_id in self.v:
            del self.v[agent_id]
        if agent_id in self.t:
            del self.t[agent_id]
