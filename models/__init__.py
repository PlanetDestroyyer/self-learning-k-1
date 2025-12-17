"""Models for K-1 Self-Learning System."""

from .k1_gpu import K1GPUModel
from .baseline_gpt import BaselineGPT

# Alias for backwards compatibility
K1Model = K1GPUModel

__all__ = ['K1GPUModel', 'K1Model', 'BaselineGPT']
