"""
Backpropagation engine for K-1 Self-Learning System.

Implements PROPER gradient computation for agent weight updates.
This is the critical fix - computing actual gradients instead of fake ones.
"""

import numpy as np
from typing import Dict, List, Tuple
from core.agent import Agent


class BackpropEngine:
    """
    Computes gradients for agent weight updates.

    Key insight: Trust decides WHICH agents to update,
    gradients decide HOW to update them.
    """

    def __init__(
        self,
        learning_rate: float = 0.0001,
        gradient_clip: float = 1.0,
        use_adam: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize backprop engine.

        Args:
            learning_rate: Base learning rate
            gradient_clip: Maximum gradient norm
            use_adam: Whether to use Adam optimizer
            beta1: Adam beta1
            beta2: Adam beta2
            epsilon: Adam epsilon
        """
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam state (per agent)
        self.m: Dict[str, Dict[str, np.ndarray]] = {}  # First moment
        self.v: Dict[str, Dict[str, np.ndarray]] = {}  # Second moment
        self.t: Dict[str, int] = {}  # Timestep

        # Statistics
        self.total_updates = 0
        self.gradient_norms = []

    def compute_agent_gradient(
        self,
        agent: Agent,
        x: np.ndarray,
        d_output: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients for an agent's weights.

        This is the CORRECT implementation:
        - x: input to the agent
        - d_output: gradient of loss w.r.t. agent's output

        Args:
            agent: Agent to compute gradients for
            x: Input vector (input_dim,)
            d_output: Upstream gradient (output_dim,)

        Returns:
            Dictionary of gradients for W1, b1, W2, b2
        """
        # Use cached hidden activation from forward pass
        h = agent.last_hidden

        if h is None:
            # Recompute if not cached
            h = np.maximum(0, x @ agent.weights['W1'] + agent.weights['b1'])

        # Gradient for output layer (W2, b2)
        # output = h @ W2 + b2
        # d_output is gradient w.r.t output
        dW2 = np.outer(h, d_output)  # (hidden_dim, output_dim)
        db2 = d_output.copy()  # (output_dim,)

        # Backprop through W2 to hidden layer
        dh = d_output @ agent.weights['W2'].T  # (hidden_dim,)

        # Backprop through ReLU
        dh_prerelu = dh.copy()
        dh_prerelu[h <= 0] = 0  # ReLU gradient: 1 if > 0, else 0

        # Gradient for input layer (W1, b1)
        # h_prerelu = x @ W1 + b1
        dW1 = np.outer(x, dh_prerelu)  # (input_dim, hidden_dim)
        db1 = dh_prerelu.copy()  # (hidden_dim,)

        return {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }

    def compute_output_layer_gradient(
        self,
        probs: np.ndarray,
        target_idx: int
    ) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss w.r.t. logits.

        For cross-entropy with softmax:
        d_loss/d_logits = probs - one_hot(target)

        Args:
            probs: Softmax probabilities (vocab_size,)
            target_idx: Index of target token

        Returns:
            Gradient w.r.t. logits (vocab_size,)
        """
        d_logits = probs.copy()
        d_logits[target_idx] -= 1.0
        return d_logits

    def clip_gradient(self, gradient: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clip gradient by global norm.

        Args:
            gradient: Dictionary of gradients

        Returns:
            Clipped gradient dictionary
        """
        # Compute global norm
        total_norm = 0.0
        for key, grad in gradient.items():
            if grad is not None:
                total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        self.gradient_norms.append(total_norm)

        # Clip if necessary
        if total_norm > self.gradient_clip:
            scale = self.gradient_clip / (total_norm + 1e-10)
            return {key: grad * scale if grad is not None else None
                    for key, grad in gradient.items()}

        return gradient

    def apply_gradient(self, agent: Agent, gradient: Dict[str, np.ndarray]):
        """
        Apply gradient to agent weights.

        Uses Adam optimizer if enabled, else vanilla SGD.

        Args:
            agent: Agent to update
            gradient: Gradient dictionary
        """
        # Clip gradient
        gradient = self.clip_gradient(gradient)

        if self.use_adam:
            self._apply_adam(agent, gradient)
        else:
            self._apply_sgd(agent, gradient)

        self.total_updates += 1

    def _apply_sgd(self, agent: Agent, gradient: Dict[str, np.ndarray]):
        """Apply vanilla SGD update."""
        for key in ['W1', 'b1', 'W2', 'b2']:
            if key in gradient and gradient[key] is not None:
                agent.weights[key] -= self.learning_rate * gradient[key]

    def _apply_adam(self, agent: Agent, gradient: Dict[str, np.ndarray]):
        """Apply Adam optimizer update."""
        agent_id = agent.id

        # Initialize Adam state if needed
        if agent_id not in self.m:
            self.m[agent_id] = {}
            self.v[agent_id] = {}
            self.t[agent_id] = 0

            for key in ['W1', 'b1', 'W2', 'b2']:
                self.m[agent_id][key] = np.zeros_like(agent.weights[key])
                self.v[agent_id][key] = np.zeros_like(agent.weights[key])

        # Increment timestep
        self.t[agent_id] += 1
        t = self.t[agent_id]

        # Update each parameter
        for key in ['W1', 'b1', 'W2', 'b2']:
            if key in gradient and gradient[key] is not None:
                g = gradient[key]

                # Update biased first moment estimate
                self.m[agent_id][key] = self.beta1 * self.m[agent_id][key] + (1 - self.beta1) * g

                # Update biased second moment estimate
                self.v[agent_id][key] = self.beta2 * self.v[agent_id][key] + (1 - self.beta2) * (g ** 2)

                # Compute bias-corrected estimates
                m_hat = self.m[agent_id][key] / (1 - self.beta1 ** t)
                v_hat = self.v[agent_id][key] / (1 - self.beta2 ** t)

                # Update weights
                agent.weights[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset_agent_state(self, agent_id: str):
        """Reset optimizer state for an agent (e.g., when agent is modified)."""
        if agent_id in self.m:
            del self.m[agent_id]
        if agent_id in self.v:
            del self.v[agent_id]
        if agent_id in self.t:
            del self.t[agent_id]

    def update_learning_rate(self, new_lr: float):
        """Update learning rate."""
        self.learning_rate = max(1e-6, min(0.1, new_lr))

    def get_statistics(self) -> Dict:
        """Get backprop statistics."""
        recent_norms = self.gradient_norms[-1000:] if self.gradient_norms else [0]

        return {
            'total_updates': self.total_updates,
            'learning_rate': self.learning_rate,
            'avg_gradient_norm': float(np.mean(recent_norms)),
            'max_gradient_norm': float(np.max(recent_norms)) if recent_norms else 0,
            'num_agents_with_state': len(self.m)
        }


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Args:
        x: Input logits

    Returns:
        Softmax probabilities
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(np.clip(x_shifted, -20, 20))
    return exp_x / (np.sum(exp_x) + 1e-10)


def cross_entropy_loss(probs: np.ndarray, target_idx: int) -> float:
    """
    Compute cross-entropy loss.

    Args:
        probs: Softmax probabilities
        target_idx: Target class index

    Returns:
        Loss value
    """
    return -np.log(np.clip(probs[target_idx], 1e-10, 1.0))
