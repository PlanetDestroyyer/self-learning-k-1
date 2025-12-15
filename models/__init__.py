"""Models for K-1 Self-Learning System."""

from .k1_model import K1SelfLearningLM
from .baseline_gpt import BaselineGPT

# Alias for backwards compatibility
K1Model = K1SelfLearningLM

__all__ = ['K1SelfLearningLM', 'K1Model', 'BaselineGPT']
