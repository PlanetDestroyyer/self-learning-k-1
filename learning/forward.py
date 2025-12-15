"""
Forward pass for K-1 Self-Learning System.

Handles forward propagation through the hierarchy.
"""

import numpy as np
from typing import Tuple, List
from core.hierarchy import Hierarchy
from core.routing import HierarchicalRouter, RoutingPath
from core.agent import Agent


class ForwardPass:
    """
    Manages forward propagation through the hierarchy.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        router: HierarchicalRouter,
        aggregation: str = 'last'
    ):
        """
        Initialize forward pass handler.

        Args:
            hierarchy: Agent hierarchy
            router: Hierarchical router
            aggregation: How to aggregate multi-agent outputs ('last', 'mean', 'weighted')
        """
        self.hierarchy = hierarchy
        self.router = router
        self.aggregation = aggregation
        self.current_iteration = 0

    def update_iteration(self, iteration: int):
        """Update current iteration."""
        self.current_iteration = iteration
        self.router.update_iteration(iteration)

    def forward(self, x: np.ndarray, mode: str = 'hard') -> Tuple[np.ndarray, RoutingPath]:
        """
        Forward pass through hierarchy.

        Args:
            x: Input vector
            mode: Routing mode ('hard' or 'soft')

        Returns:
            (output, routing_path)
        """
        return self.router.route(x, mode)

    def forward_batch(
        self,
        batch: np.ndarray,
        mode: str = 'hard'
    ) -> Tuple[np.ndarray, List[RoutingPath]]:
        """
        Forward pass for a batch of inputs.

        Args:
            batch: Input batch (batch_size, input_dim)
            mode: Routing mode

        Returns:
            (outputs, routing_paths)
        """
        outputs = []
        paths = []

        for x in batch:
            output, path = self.forward(x, mode)
            outputs.append(output)
            paths.append(path)

        return np.array(outputs), paths

    def get_activated_agents(self, path: RoutingPath) -> List[Agent]:
        """Get all agents activated in a routing path."""
        return path.get_activated_agents()

    def compute_output_with_projection(
        self,
        hidden: np.ndarray,
        output_projection: np.ndarray
    ) -> np.ndarray:
        """
        Project hidden state to output (e.g., vocabulary logits).

        Args:
            hidden: Hidden state from hierarchy
            output_projection: Projection matrix (hidden_dim, output_dim)

        Returns:
            Output logits
        """
        return hidden @ output_projection
