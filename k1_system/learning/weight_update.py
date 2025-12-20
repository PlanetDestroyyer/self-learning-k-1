"""
Weight update system for the Self-Learning K-1 System (PyTorch version).

Uses PyTorch optimizers and autograd instead of manual gradient computation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from ..core.agent import Agent


class WeightUpdater:
    """
    Manages weight updates for agents using PyTorch optimizers.

    This replaces manual gradient computation with PyTorch's autograd.
    Each agent gets its own optimizer to maintain separate momentum states.
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize weight updater.

        Args:
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.update_count = 0

        # Store one SGD optimizer per agent
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

    def get_optimizer(self, agent: Agent) -> torch.optim.Optimizer:
        """
        Get or create optimizer for an agent.

        Args:
            agent: Agent to get optimizer for

        Returns:
            PyTorch optimizer for this agent
        """
        if agent.id not in self.optimizers:
            self.optimizers[agent.id] = torch.optim.SGD(
                agent.parameters(),
                lr=self.learning_rate
            )
        return self.optimizers[agent.id]

    def update_agent(self, agent: Agent):
        """
        Update an agent's weights using PyTorch optimizer.

        NOTE: Gradients must be computed via loss.backward() BEFORE calling this.

        Args:
            agent: Agent to update
        """
        optimizer = self.get_optimizer(agent)

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Zero gradients for next iteration
        optimizer.zero_grad()

        self.update_count += 1

    def update_agents(self, agents: List[Agent]):
        """
        Update multiple agents with their gradients.

        Args:
            agents: List of agents to update
        """
        for agent in agents:
            self.update_agent(agent)

    def update_learning_rate(self, new_rate: float):
        """
        Update learning rate for all optimizers.

        Args:
            new_rate: New learning rate
        """
        self.learning_rate = max(0.0, new_rate)

        # Update all existing optimizers
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    def get_gradient_norm(self, agent: Agent) -> float:
        """
        Compute L2 norm of agent's gradients.

        Args:
            agent: Agent to compute gradient norm for

        Returns:
            Gradient norm
        """
        total_norm = 0.0
        for param in agent.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def zero_grad_all(self):
        """Zero gradients for all agents."""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def remove_agent_optimizer(self, agent_id: str):
        """
        Remove optimizer for a deleted agent (for pruning).

        Args:
            agent_id: Agent ID to remove
        """
        if agent_id in self.optimizers:
            del self.optimizers[agent_id]

    def get_update_statistics(self) -> Dict:
        """
        Get statistics about weight updates.

        Returns:
            Dictionary of statistics
        """
        return {
            'total_updates': self.update_count,
            'current_learning_rate': self.learning_rate,
            'num_optimizers': len(self.optimizers)
        }


class AdaptiveWeightUpdater(WeightUpdater):
    """
    Weight updater with adaptive learning rate (Adam optimizer).

    Uses PyTorch's Adam optimizer instead of manual Adam implementation.
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

        # Adam optimizers for each agent
        self.optimizers: Dict[str, torch.optim.Adam] = {}

    def get_optimizer(self, agent: Agent) -> torch.optim.Adam:
        """
        Get or create Adam optimizer for an agent.

        Args:
            agent: Agent to get optimizer for

        Returns:
            PyTorch Adam optimizer for this agent
        """
        if agent.id not in self.optimizers:
            self.optimizers[agent.id] = torch.optim.Adam(
                agent.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.epsilon
            )
        return self.optimizers[agent.id]

    def reset_agent_state(self, agent_id: str):
        """
        Reset optimizer state for an agent.

        Args:
            agent_id: Agent ID
        """
        if agent_id in self.optimizers:
            del self.optimizers[agent_id]
