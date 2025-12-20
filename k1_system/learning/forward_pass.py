"""
Forward pass implementation for the Self-Learning K-1 System.

Implements hierarchical forward propagation through the agent tree.
"""

import numpy as np
import torch
from typing import Tuple, Union
from ..core.hierarchy import Hierarchy
from ..core.routing import HierarchicalRouter, RoutingPath
from ..core.trust_system import TrustCache


class ForwardPass:
    """
    Manages forward propagation through the hierarchical network.
    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 router: HierarchicalRouter,
                 trust_cache: TrustCache):
        """
        Initialize forward pass system.

        Args:
            hierarchy: Hierarchy structure
            router: Router for hierarchical routing
            trust_cache: Trust cache for quick specialist lookup
        """
        self.hierarchy = hierarchy
        self.router = router
        self.trust_cache = trust_cache

        self.current_iteration = 0

    def forward(self,
               x: np.ndarray,
               mode: str = 'hard',
               use_cache: bool = False,
               error_type: str = None) -> Tuple[np.ndarray, RoutingPath]:
        """
        Perform forward pass through hierarchy.

        Args:
            x: Input vector
            mode: Routing mode ('hard' or 'soft')
            use_cache: Whether to use trust cache
            error_type: Type of error (for cache lookup)

        Returns:
            (output, routing_path) tuple
        """
        # Route through hierarchy
        if use_cache and error_type:
            path = self.router.route_with_cache(x, self.trust_cache, error_type)
        else:
            path = self.router.route(x, mode=mode)

        # Update iteration count for all activated agents
        for agent in path.get_activated_agents():
            agent.last_used = self.current_iteration

        # Get final output
        output = path.final_output

        if output is None:
            # No output generated, use root output
            output = self.hierarchy.root.forward(x)

        return output, path

    def batch_forward(self,
                     X: Union[np.ndarray, torch.Tensor],
                     mode: str = 'hard',
                     vectorized: bool = True) -> Tuple[Union[np.ndarray, torch.Tensor], list]:
        """
        Perform forward pass on a batch of inputs (vectorized).

        Args:
            X: Batch of input vectors (batch_size x input_dim)
            mode: Routing mode
            vectorized: If True, uses vectorized batch processing (faster but all samples follow same path)

        Returns:
            (outputs, paths) tuple
        """
        # Convert to tensor if needed
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X = torch.from_numpy(X).float()

        batch_size = X.shape[0]

        if vectorized and batch_size > 1:
            # Vectorized version: route first sample, apply to all
            # Trade-off: All samples follow same agent path, but much faster

            # Get path from first sample
            first_sample = X[0].cpu().numpy() if isinstance(X, torch.Tensor) else X[0]
            _, sample_path = self.forward(first_sample, mode=mode)
            agents_in_path = sample_path.get_activated_agents()

            # Apply same path to entire batch (vectorized)
            current_output = X
            for agent in agents_in_path:
                current_output = agent.forward(current_output)
                # Update agent usage
                agent.last_used = self.current_iteration

            # Create paths (all use same structure)
            paths = [sample_path for _ in range(batch_size)]
            outputs = current_output

        else:
            # Sequential version: route each sample independently
            outputs = []
            paths = []

            for i in range(batch_size):
                x_i = X[i].cpu().numpy() if isinstance(X, torch.Tensor) else X[i]
                output, path = self.forward(x_i, mode=mode)
                outputs.append(output)
                paths.append(path)

            # Stack outputs
            if isinstance(outputs[0], torch.Tensor):
                outputs = torch.stack(outputs)
            else:
                outputs = np.array(outputs)

        # Convert back to numpy if input was numpy
        if is_numpy and isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().detach().numpy()

        return outputs, paths

    def compute_prediction(self, output: np.ndarray, task: str = 'classification') -> np.ndarray:
        """
        Convert network output to prediction.

        Args:
            output: Raw network output
            task: Task type ('classification' or 'regression')

        Returns:
            Prediction
        """
        if task == 'classification':
            # Apply softmax
            exp_output = np.exp(output - np.max(output))
            probabilities = exp_output / np.sum(exp_output)
            return probabilities
        elif task == 'regression':
            # Direct output
            return output
        else:
            raise ValueError(f"Unknown task type: {task}")

    def compute_loss(self,
                    output: np.ndarray,
                    target: np.ndarray,
                    task: str = 'classification') -> float:
        """
        Compute loss between output and target.

        Args:
            output: Network output
            target: Target output
            task: Task type

        Returns:
            Loss value
        """
        if task == 'classification':
            # Cross-entropy loss
            prediction = self.compute_prediction(output, task='classification')
            # Avoid log(0)
            prediction = np.clip(prediction, 1e-10, 1.0)
            if len(target.shape) == 1 or target.shape[0] == 1:
                # Single target class
                loss = -np.log(prediction[int(target)])
            else:
                # One-hot target
                loss = -np.sum(target * np.log(prediction))
            return loss
        elif task == 'regression':
            # Mean squared error
            loss = 0.5 * np.sum((output - target) ** 2)
            return loss
        else:
            raise ValueError(f"Unknown task type: {task}")

    def compute_accuracy(self,
                        outputs: np.ndarray,
                        targets: np.ndarray,
                        task: str = 'classification') -> float:
        """
        Compute accuracy for a batch.

        Args:
            outputs: Batch of network outputs
            targets: Batch of targets
            task: Task type

        Returns:
            Accuracy (0.0 to 1.0)
        """
        if task == 'classification':
            predictions = []
            for output in outputs:
                pred = self.compute_prediction(output, task='classification')
                predictions.append(np.argmax(pred))

            if len(targets.shape) == 1:
                # Targets are class indices
                correct = np.sum(np.array(predictions) == targets)
            else:
                # Targets are one-hot
                correct = np.sum(np.array(predictions) == np.argmax(targets, axis=1))

            return correct / len(targets)
        elif task == 'regression':
            # For regression, compute R^2 score or similar
            # For simplicity, return MSE-based score
            mse = np.mean((outputs - targets) ** 2)
            return 1.0 / (1.0 + mse)  # Convert to 0-1 score
        else:
            raise ValueError(f"Unknown task type: {task}")

    def update_iteration(self, iteration: int):
        """Update current iteration count."""
        self.current_iteration = iteration

    def get_activated_agents_stats(self, paths: list) -> dict:
        """
        Get statistics about activated agents.

        Args:
            paths: List of routing paths

        Returns:
            Dictionary of statistics
        """
        if not paths:
            return {}

        all_activated = set()
        total_activations = 0
        path_lengths = []

        for path in paths:
            agents = path.get_activated_agents()
            all_activated.update([a.id for a in agents])
            total_activations += len(agents)
            path_lengths.append(path.get_path_length())

        return {
            'unique_agents_activated': len(all_activated),
            'total_activations': total_activations,
            'avg_activations_per_sample': total_activations / len(paths),
            'avg_path_length': np.mean(path_lengths),
            'max_path_length': np.max(path_lengths),
            'min_path_length': np.min(path_lengths)
        }
