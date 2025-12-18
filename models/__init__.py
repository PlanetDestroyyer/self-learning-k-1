"""Models for K-1 Self-Learning System."""

from .k1_gpu import K1GPUModel
from .baseline_gpt_pytorch import BaselineGPTPyTorch

# Alias for backwards compatibility
K1Model = K1GPUModel
BaselineGPT = BaselineGPTPyTorch  # Backwards compatibility alias

__all__ = ['K1GPUModel', 'K1Model', 'BaselineGPTPyTorch', 'BaselineGPT']

