"""
Trust system for K-1 Self-Learning System.

Manages trust scores and trust cache.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from .agent import Agent


class TrustCache:
    """Cache for high-trust agents."""

    def __init__(self, cache_threshold: float = 0.7):
        """
        Initialize trust cache.

        Args:
            cache_threshold: Minimum trust to be cached
        """
        self.cache_threshold = cache_threshold
        self.cache: Dict[str, Dict] = {}
        self.hits = 0
        self.misses = 0

    def add_to_cache(self, agent: Agent):
        """Add agent to cache if trust high enough."""
        if agent.trust >= self.cache_threshold:
            self.cache[agent.id] = {
                'specialty': agent.specialty,
                'trust': agent.trust,
                'success_count': agent.success_count,
                'avg_error_reduction': agent.total_error_reduction / max(1, agent.success_count),
                'last_updated': agent.last_used
            }

    def get_cached(self, agent_id: str) -> Optional[Dict]:
        """Get cached info for agent."""
        if agent_id in self.cache:
            self.hits += 1
            return self.cache[agent_id]
        self.misses += 1
        return None

    def remove_from_cache(self, agent_id: str):
        """Remove agent from cache."""
        if agent_id in self.cache:
            del self.cache[agent_id]

    def get_cache_size(self) -> int:
        """Get number of cached agents."""
        return len(self.cache)

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def decay_cache(self, current_iteration: int, decay_window: int = 10000):
        """Decay old cache entries."""
        to_remove = []
        for agent_id, info in self.cache.items():
            if current_iteration - info['last_updated'] > decay_window:
                info['trust'] *= 0.9
                if info['trust'] < self.cache_threshold:
                    to_remove.append(agent_id)

        for agent_id in to_remove:
            del self.cache[agent_id]

    def update_threshold(self, new_threshold: float):
        """Update cache threshold."""
        self.cache_threshold = max(0.0, min(1.0, new_threshold))

        # Remove entries below new threshold
        to_remove = [aid for aid, info in self.cache.items() if info['trust'] < new_threshold]
        for agent_id in to_remove:
            del self.cache[agent_id]


class TrustSystem:
    """
    Manages trust scoring and updates for agents.
    """

    def __init__(
        self,
        error_penalty_multiplier: float = 0.95,
        success_reward_multiplier: float = 0.3,
        success_reward_cap: float = 0.2,
        cache_threshold: float = 0.7,
        min_trust: float = 0.0,
        max_trust: float = 1.0
    ):
        """
        Initialize trust system.

        Args:
            error_penalty_multiplier: Multiply trust by this on error
            success_reward_multiplier: Scale success reward by this
            success_reward_cap: Maximum trust increase per success
            cache_threshold: Threshold for trust cache
            min_trust: Minimum trust value
            max_trust: Maximum trust value
        """
        self.error_penalty_multiplier = error_penalty_multiplier
        self.success_reward_multiplier = success_reward_multiplier
        self.success_reward_cap = success_reward_cap
        self.min_trust = min_trust
        self.max_trust = max_trust

        self.trust_cache = TrustCache(cache_threshold)
        self.trust_history: Dict[str, List[float]] = defaultdict(list)

    def report_success(self, agent: Agent, error_reduction: float):
        """
        Report that agent helped reduce error.

        Args:
            agent: Agent that succeeded
            error_reduction: Amount of error reduction
        """
        reward = min(error_reduction * self.success_reward_multiplier, self.success_reward_cap)
        agent.trust = min(self.max_trust, agent.trust + reward)
        agent.success_count += 1
        agent.total_error_reduction += error_reduction

        self.trust_history[agent.id].append(agent.trust)

        # Update cache
        self.trust_cache.add_to_cache(agent)

    def report_error(self, agent: Agent, error_magnitude: float):
        """
        Report that agent contributed to error.

        Args:
            agent: Agent that failed
            error_magnitude: Magnitude of error
        """
        # Scale penalty by error magnitude
        penalty_scale = min(1.0, error_magnitude)
        penalty = 1.0 - (1.0 - self.error_penalty_multiplier) * penalty_scale

        agent.trust = max(self.min_trust, agent.trust * penalty)
        agent.failure_count += 1
        agent.record_error(error_magnitude)

        self.trust_history[agent.id].append(agent.trust)

        # Remove from cache if trust dropped below threshold
        if agent.trust < self.trust_cache.cache_threshold:
            self.trust_cache.remove_from_cache(agent.id)

    def compute_avg_trust(self, agents: List[Agent]) -> float:
        """Compute average trust across agents."""
        if not agents:
            return 0.0
        return float(np.mean([a.trust for a in agents]))

    def compute_trust_variance(self, agents: List[Agent]) -> float:
        """Compute trust variance across agents."""
        if not agents:
            return 0.0
        return float(np.var([a.trust for a in agents]))

    def get_trust_distribution(self, agents: List[Agent]) -> Dict:
        """Get trust distribution statistics."""
        if not agents:
            return {}

        trusts = [a.trust for a in agents]

        return {
            'mean': float(np.mean(trusts)),
            'std': float(np.std(trusts)),
            'min': float(np.min(trusts)),
            'max': float(np.max(trusts)),
            'median': float(np.median(trusts)),
            'below_0.3': sum(1 for t in trusts if t < 0.3),
            'above_0.7': sum(1 for t in trusts if t > 0.7)
        }

    def get_top_agents(self, agents: List[Agent], k: int = 10) -> List[Agent]:
        """Get top-k agents by trust."""
        return sorted(agents, key=lambda a: a.trust, reverse=True)[:k]

    def get_bottom_agents(self, agents: List[Agent], k: int = 10) -> List[Agent]:
        """Get bottom-k agents by trust."""
        return sorted(agents, key=lambda a: a.trust)[:k]

    def update_parameters(
        self,
        error_penalty: float = None,
        success_reward: float = None,
        reward_cap: float = None
    ):
        """Update trust system parameters."""
        if error_penalty is not None:
            self.error_penalty_multiplier = max(0.5, min(1.0, error_penalty))
        if success_reward is not None:
            self.success_reward_multiplier = max(0.0, min(1.0, success_reward))
        if reward_cap is not None:
            self.success_reward_cap = max(0.0, min(0.5, reward_cap))

    def decay_all_trusts(self, agents: List[Agent], decay_rate: float = 0.99):
        """Apply small decay to all trusts (use sparingly)."""
        for agent in agents:
            agent.trust = max(self.min_trust, agent.trust * decay_rate)
