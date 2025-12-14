"""
Trust system for the Self-Learning K-1 System.

Manages trust scores and trust cache for agents.
Trust drives credit assignment and structural decisions.
"""

from typing import Dict, List, Optional
import numpy as np
from .agent import Agent


class TrustCache:
    """
    Cache for storing proven reliable agents.

    Agents with high trust are added to cache for quick lookup
    when similar errors occur.
    """

    def __init__(self, cache_threshold: float = 0.7):
        """
        Initialize trust cache.

        Args:
            cache_threshold: Minimum trust to be added to cache
        """
        self.cache_threshold = cache_threshold
        self.cache: Dict[str, Dict] = {}

    def add_to_cache(self, agent: Agent):
        """
        Add an agent to the trust cache.

        Args:
            agent: Agent to add
        """
        if agent.trust >= self.cache_threshold:
            self.cache[agent.id] = {
                'trust_score': agent.trust,
                'specialization': agent.specialty,
                'error_types_solved': [],
                'success_count': agent.success_count,
                'failure_count': agent.failure_count,
                'avg_error_reduction': (agent.total_error_reduction /
                                       max(agent.success_count, 1)),
                'last_used': agent.last_used,
                'agent_ref': agent
            }

    def remove_from_cache(self, agent_id: str):
        """Remove an agent from cache."""
        if agent_id in self.cache:
            del self.cache[agent_id]

    def get_cached_agent(self, agent_id: str) -> Optional[Dict]:
        """Get cached agent info."""
        return self.cache.get(agent_id)

    def get_specialists_for_error_type(self, error_type: str) -> List[Agent]:
        """
        Get specialists for a specific error type from cache.

        Args:
            error_type: Type of error

        Returns:
            List of specialist agents
        """
        specialists = []
        for agent_id, info in self.cache.items():
            if error_type in info['error_types_solved']:
                specialists.append(info['agent_ref'])

        # Sort by trust score
        specialists.sort(key=lambda a: a.trust, reverse=True)
        return specialists

    def decay_cache(self, current_iteration: int, window: int = 10000):
        """
        Decay trust of agents not recently used.

        Args:
            current_iteration: Current iteration
            window: Window for considering "recent"
        """
        to_remove = []

        for agent_id, info in self.cache.items():
            if current_iteration - info['last_used'] > window:
                # Agent not recently used, decay trust
                agent = info['agent_ref']
                agent.trust *= 0.1

                # Remove from cache if trust too low
                if agent.trust < self.cache_threshold:
                    to_remove.append(agent_id)
                else:
                    # Update cache entry
                    info['trust_score'] = agent.trust

        # Remove decayed agents
        for agent_id in to_remove:
            self.remove_from_cache(agent_id)

    def get_cache_size(self) -> int:
        """Get number of agents in cache."""
        return len(self.cache)

    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        if not self.cache:
            return {
                'size': 0,
                'avg_trust': 0.0,
                'avg_success_rate': 0.0
            }

        trust_scores = [info['trust_score'] for info in self.cache.values()]
        success_rates = []

        for info in self.cache.values():
            total = info['success_count'] + info['failure_count']
            if total > 0:
                success_rates.append(info['success_count'] / total)

        return {
            'size': len(self.cache),
            'avg_trust': np.mean(trust_scores),
            'avg_success_rate': np.mean(success_rates) if success_rates else 0.0,
            'max_trust': np.max(trust_scores),
            'min_trust': np.min(trust_scores)
        }


class TrustSystem:
    """
    Manages trust scores and updates for all agents.
    """

    def __init__(self,
                 error_penalty_multiplier: float = 0.9,
                 success_reward_multiplier: float = 0.5,
                 success_reward_cap: float = 0.3,
                 cache_threshold: float = 0.7):
        """
        Initialize trust system.

        Args:
            error_penalty_multiplier: Multiplier for trust penalty on error
            success_reward_multiplier: Multiplier for trust reward on success
            success_reward_cap: Maximum trust reward per success
            cache_threshold: Threshold for adding to trust cache
        """
        self.error_penalty_multiplier = error_penalty_multiplier
        self.success_reward_multiplier = success_reward_multiplier
        self.success_reward_cap = success_reward_cap

        self.trust_cache = TrustCache(cache_threshold)
        self.error_history: Dict[str, List[float]] = {}

    def report_error(self, agent: Agent, error_magnitude: float):
        """
        Report that an agent contributed to an error.

        Args:
            agent: Agent that made error
            error_magnitude: Magnitude of error (0.0 to 1.0+)
        """
        # Apply trust penalty
        agent.trust *= self.error_penalty_multiplier

        # Ensure trust stays in valid range
        agent.trust = max(0.0, min(1.0, agent.trust))

        # Record error
        agent.record_error(error_magnitude)
        agent.failure_count += 1

        # Track in error history
        if agent.id not in self.error_history:
            self.error_history[agent.id] = []
        self.error_history[agent.id].append(error_magnitude)

        # Remove from cache if trust too low
        if agent.trust < self.trust_cache.cache_threshold:
            self.trust_cache.remove_from_cache(agent.id)

    def report_success(self, agent: Agent, error_reduction: float):
        """
        Report that an agent successfully reduced error.

        Args:
            agent: Agent that succeeded
            error_reduction: Amount of error reduced (0.0 to 1.0+)
        """
        # Calculate reward
        reward = min(error_reduction * self.success_reward_multiplier,
                    self.success_reward_cap)

        # Apply trust reward
        agent.trust = min(1.0, agent.trust + reward)

        # Record success
        agent.success_count += 1
        agent.total_error_reduction += error_reduction

        # Add to cache if trust high enough
        if agent.trust >= self.trust_cache.cache_threshold:
            self.trust_cache.add_to_cache(agent)

    def get_agents_below_trust(self, agents: List[Agent], threshold: float) -> List[Agent]:
        """
        Get agents with trust below threshold.

        Args:
            agents: List of agents to check
            threshold: Trust threshold

        Returns:
            Agents below threshold
        """
        return [agent for agent in agents if agent.trust < threshold]

    def get_top_trusted_agents(self, agents: List[Agent], k: int) -> List[Agent]:
        """
        Get top k most trusted agents.

        Args:
            agents: List of agents
            k: Number to return

        Returns:
            Top k agents by trust
        """
        sorted_agents = sorted(agents, key=lambda a: a.trust, reverse=True)
        return sorted_agents[:k]

    def compute_trust_variance(self, agents: List[Agent]) -> float:
        """
        Compute variance in trust scores.

        Args:
            agents: List of agents

        Returns:
            Variance in trust scores
        """
        if not agents:
            return 0.0

        trust_scores = [agent.trust for agent in agents]
        return np.var(trust_scores)

    def compute_avg_trust(self, agents: List[Agent]) -> float:
        """
        Compute average trust score.

        Args:
            agents: List of agents

        Returns:
            Average trust
        """
        if not agents:
            return 0.0

        trust_scores = [agent.trust for agent in agents]
        return np.mean(trust_scores)

    def rank_agents_by_responsibility(self,
                                     agents: List[Agent],
                                     error_magnitude: float) -> List[tuple]:
        """
        Rank agents by responsibility for an error.

        Args:
            agents: List of agents that were activated
            error_magnitude: Magnitude of error

        Returns:
            List of (agent, ranking_score) tuples, sorted by score
        """
        rankings = []

        for agent in agents:
            # Compute responsibility
            responsibility = agent.compute_responsibility(error_magnitude)

            # Compute ranking score (trusted agents get priority)
            ranking_score = agent.compute_ranking_score(responsibility)

            rankings.append((agent, ranking_score))

        # Sort by ranking score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def select_top_k_for_update(self,
                                agents: List[Agent],
                                error_magnitude: float,
                                k: int) -> List[Agent]:
        """
        Select top k agents to update based on trust-weighted responsibility.

        Args:
            agents: List of activated agents
            error_magnitude: Magnitude of error
            k: Number of agents to select

        Returns:
            Top k agents for update
        """
        rankings = self.rank_agents_by_responsibility(agents, error_magnitude)

        # Select top k
        top_k = [agent for agent, score in rankings[:k]]

        return top_k

    def get_trust_distribution(self, agents: List[Agent]) -> Dict:
        """
        Get distribution of trust scores.

        Args:
            agents: List of agents

        Returns:
            Dictionary with distribution statistics
        """
        if not agents:
            return {}

        trust_scores = np.array([agent.trust for agent in agents])

        return {
            'mean': float(np.mean(trust_scores)),
            'std': float(np.std(trust_scores)),
            'min': float(np.min(trust_scores)),
            'max': float(np.max(trust_scores)),
            'median': float(np.median(trust_scores)),
            'q25': float(np.percentile(trust_scores, 25)),
            'q75': float(np.percentile(trust_scores, 75)),
            'below_0.2': int(np.sum(trust_scores < 0.2)),
            'above_0.7': int(np.sum(trust_scores > 0.7))
        }

    def decay_cache(self, current_iteration: int, window: int = 10000):
        """
        Decay trust cache for unused agents.

        Args:
            current_iteration: Current iteration
            window: Window for considering "recent"
        """
        self.trust_cache.decay_cache(current_iteration, window)

    def get_cache_hit_rate(self, lookups: int, hits: int) -> float:
        """
        Compute cache hit rate.

        Args:
            lookups: Total cache lookups
            hits: Successful hits

        Returns:
            Hit rate (0.0 to 1.0)
        """
        if lookups == 0:
            return 0.0
        return hits / lookups
