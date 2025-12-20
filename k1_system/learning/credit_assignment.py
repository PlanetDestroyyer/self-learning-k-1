"""
Credit assignment for the Self-Learning K-1 System (PyTorch version).

Uses gradient-based responsibility + trust for agent selection.
Computes real gradients via PyTorch autograd, then selects top-K agents.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
import numpy as np
from ..core.agent import Agent
from ..core.trust_system import TrustSystem
from ..core.routing import RoutingPath


class CreditAssignmentSystem:
    """
    Manages credit assignment to agents based on trust and responsibility.
    """

    def __init__(self,
                 trust_system: TrustSystem,
                 top_k: int = 3):
        """
        Initialize credit assignment system.

        Args:
            trust_system: Trust system for managing trust scores
            top_k: Number of top agents to select for update
        """
        self.trust_system = trust_system
        self.top_k = top_k

    def assign_credit(self,
                     routing_path: RoutingPath,
                     target: Union[torch.Tensor, np.ndarray],
                     prediction: Union[torch.Tensor, np.ndarray]) -> Tuple[List[Agent], torch.Tensor]:
        """
        Assign credit to agents using PyTorch gradients.

        Computes loss, runs backward pass, then selects top-K agents by gradient magnitude.

        Args:
            routing_path: Path taken during forward pass
            target: Target output (tensor or numpy)
            prediction: Predicted output (tensor or numpy)

        Returns:
            Tuple of (selected_agents, loss_value)
        """
        # Convert to tensors if needed
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        if isinstance(prediction, np.ndarray):
            prediction = torch.from_numpy(prediction).float()

        # Move to same device as prediction
        if target.device != prediction.device:
            target = target.to(prediction.device)

        # Compute loss (MSE for now)
        loss = F.mse_loss(prediction, target)

        # Backward pass to compute gradients
        loss.backward(retain_graph=True)

        # Get activated agents
        activated_agents = routing_path.get_activated_agents()

        if not activated_agents:
            return [], loss

        # Compute gradient magnitudes (our "responsibility")
        gradient_magnitudes = self._compute_gradient_magnitudes(activated_agents)

        # Rank agents by gradient magnitude weighted by trust
        ranked_agents = self._rank_by_trust_weighted_responsibility(
            activated_agents, gradient_magnitudes
        )

        # Select top-K agents
        selected_agents = self._select_top_k(ranked_agents)

        return selected_agents, loss

    def _compute_gradient_magnitudes(self, agents: List[Agent]) -> List[float]:
        """
        Compute gradient magnitude for each agent (PyTorch version).

        Gradient magnitude = sum of L2 norms of all parameter gradients.

        Args:
            agents: List of activated agents

        Returns:
            List of gradient magnitudes
        """
        gradient_magnitudes = []

        for agent in agents:
            total_grad_norm = 0.0
            for param in agent.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2

            # L2 norm
            gradient_magnitudes.append(total_grad_norm ** 0.5)

        return gradient_magnitudes

    def _compute_responsibilities(self,
                                 agents: List[Agent],
                                 error_magnitude: float) -> List[float]:
        """
        Legacy method: Compute responsibility score for each agent.

        NOTE: In PyTorch version, use _compute_gradient_magnitudes instead.

        Args:
            agents: List of activated agents
            error_magnitude: Magnitude of error

        Returns:
            List of responsibility scores
        """
        responsibilities = []

        for agent in agents:
            responsibility = agent.compute_responsibility(error_magnitude)
            responsibilities.append(responsibility)

        return responsibilities

    def _rank_by_trust_weighted_responsibility(self,
                                               agents: List[Agent],
                                               responsibilities: List[float]) -> List[Tuple[Agent, float]]:
        """
        Rank agents by trust-weighted responsibility.

        Ranking_score = responsibility * (1.0 + trust)

        Trusted agents get priority for updates.

        Args:
            agents: List of agents
            responsibilities: List of responsibility scores

        Returns:
            List of (agent, ranking_score) tuples, sorted by score
        """
        rankings = []

        for agent, responsibility in zip(agents, responsibilities):
            ranking_score = agent.compute_ranking_score(responsibility)
            rankings.append((agent, ranking_score))

        # Sort by ranking score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def _select_top_k(self, ranked_agents: List[Tuple[Agent, float]]) -> List[Agent]:
        """
        Select top-K agents from ranked list.

        Args:
            ranked_agents: List of (agent, score) tuples

        Returns:
            Top-K agents
        """
        # Get top K
        top_k = min(self.top_k, len(ranked_agents))
        selected = [agent for agent, score in ranked_agents[:top_k]]

        return selected

    def update_top_k(self, new_top_k: int):
        """
        Update the top-K parameter.

        Args:
            new_top_k: New value for top-K
        """
        self.top_k = max(1, new_top_k)  # At least 1

    def compute_error_reduction(self,
                               old_error: float,
                               new_error: float) -> float:
        """
        Compute error reduction after updates.

        Args:
            old_error: Error before update
            new_error: Error after update

        Returns:
            Error reduction (positive = improvement)
        """
        return old_error - new_error

    def assign_rewards_and_penalties(self,
                                    updated_agents: List[Agent],
                                    error_reduction: float,
                                    old_error: float):
        """
        Assign rewards/penalties based on update results.

        Args:
            updated_agents: Agents that were updated
            error_reduction: Change in error (positive = improvement)
            old_error: Original error magnitude
        """
        if error_reduction > 0:
            # Improvement! Reward agents
            for agent in updated_agents:
                self.trust_system.report_success(agent, error_reduction)
        else:
            # Made it worse! Penalize agents
            for agent in updated_agents:
                self.trust_system.report_error(agent, abs(error_reduction))

    def get_credit_statistics(self,
                             agents: List[Agent],
                             error_magnitude: float) -> dict:
        """
        Get statistics about credit distribution.

        Args:
            agents: List of agents
            error_magnitude: Error magnitude

        Returns:
            Dictionary of statistics
        """
        if not agents:
            return {}

        responsibilities = self._compute_responsibilities(agents, error_magnitude)
        rankings = self._rank_by_trust_weighted_responsibility(agents, responsibilities)

        ranking_scores = [score for _, score in rankings]

        return {
            'num_agents': len(agents),
            'total_responsibility': sum(responsibilities),
            'avg_responsibility': np.mean(responsibilities),
            'max_responsibility': max(responsibilities) if responsibilities else 0,
            'avg_ranking_score': np.mean(ranking_scores) if ranking_scores else 0,
            'top_k_selected': min(self.top_k, len(agents)),
            'selection_ratio': min(self.top_k, len(agents)) / len(agents) if agents else 0
        }
