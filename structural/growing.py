"""
Self-growing system for K-1 Self-Learning System.

Creates new specialist agents for knowledge gaps.
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from core.agent import Agent
from core.hierarchy import Hierarchy


class ErrorCluster:
    """Represents a cluster of similar errors."""

    def __init__(self, cluster_id: int, centroid: np.ndarray):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.error_count = 0
        self.avg_magnitude = 0.0
        self.first_seen = None
        self.last_seen = None
        self.magnitudes = []

    def add_error(self, magnitude: float, iteration: int):
        """Add an error to this cluster."""
        self.magnitudes.append(magnitude)
        self.error_count += 1
        self.avg_magnitude = np.mean(self.magnitudes[-1000:])  # Recent avg
        self.last_seen = iteration

        if self.first_seen is None:
            self.first_seen = iteration

    def persistence(self, current_iteration: int) -> int:
        """How long has this cluster persisted?"""
        if self.first_seen is None:
            return 0
        return current_iteration - self.first_seen


class GrowingSystem:
    """
    Manages autonomous creation of new agents.

    New agents are created when:
    - Persistent error cluster detected
    - No existing agent handles it well
    - Room in hierarchy for more agents
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        error_cluster_min_size: int = 100,
        persistence_threshold: int = 5000,
        max_agents: int = 100,
        new_agent_trust: float = 0.3,
        validation_period: int = 1000
    ):
        """
        Initialize growing system.

        Args:
            hierarchy: Hierarchy to grow
            error_cluster_min_size: Minimum errors in cluster
            persistence_threshold: How long cluster must persist
            max_agents: Maximum total agents
            new_agent_trust: Initial trust for new agents
            validation_period: Iterations to validate new agent
        """
        self.hierarchy = hierarchy
        self.error_cluster_min_size = error_cluster_min_size
        self.persistence_threshold = persistence_threshold
        self.max_agents = max_agents
        self.new_agent_trust = new_agent_trust
        self.validation_period = validation_period

        # Error tracking
        self.error_history: List[Dict] = []
        self.error_clusters: Dict[int, ErrorCluster] = {}
        self.next_cluster_id = 0

        # Agents under validation
        self.validating_agents: List[Dict] = []

        # History
        self.creation_history = []
        self.total_created = 0

    def record_error(
        self,
        error_vector: np.ndarray,
        error_magnitude: float,
        activated_agents: List[Agent],
        iteration: int
    ):
        """
        Record an error for gap detection.

        Args:
            error_vector: Error direction/pattern
            error_magnitude: Error size
            activated_agents: Agents that were activated
            iteration: Current iteration
        """
        # Get best agent trust for this error
        best_trust = max([a.trust for a in activated_agents]) if activated_agents else 0

        self.error_history.append({
            'vector': error_vector,
            'magnitude': error_magnitude,
            'best_trust': best_trust,
            'iteration': iteration
        })

        # Keep history bounded
        if len(self.error_history) > 50000:
            self.error_history = self.error_history[-25000:]

        # Assign to cluster (simple nearest-centroid)
        self._assign_to_cluster(error_vector, error_magnitude, iteration)

    def _assign_to_cluster(
        self,
        error_vector: np.ndarray,
        magnitude: float,
        iteration: int
    ):
        """Assign error to nearest cluster or create new one."""
        if len(self.error_clusters) == 0:
            # Create first cluster
            cluster = ErrorCluster(self.next_cluster_id, error_vector.copy())
            cluster.add_error(magnitude, iteration)
            self.error_clusters[self.next_cluster_id] = cluster
            self.next_cluster_id += 1
            return

        # Find nearest cluster
        best_cluster = None
        best_distance = float('inf')

        for cluster in self.error_clusters.values():
            dist = np.linalg.norm(error_vector - cluster.centroid)
            if dist < best_distance:
                best_distance = dist
                best_cluster = cluster

        # If close enough, add to cluster
        threshold = np.linalg.norm(error_vector) * 0.5  # 50% of error norm

        if best_distance < threshold:
            best_cluster.add_error(magnitude, iteration)
            # Update centroid (running average)
            alpha = 0.01
            best_cluster.centroid = (1 - alpha) * best_cluster.centroid + alpha * error_vector
        else:
            # Create new cluster
            if len(self.error_clusters) < 50:  # Limit clusters
                cluster = ErrorCluster(self.next_cluster_id, error_vector.copy())
                cluster.add_error(magnitude, iteration)
                self.error_clusters[self.next_cluster_id] = cluster
                self.next_cluster_id += 1

    def is_knowledge_gap(self, cluster: ErrorCluster, current_iteration: int) -> bool:
        """
        Determine if a cluster represents a knowledge gap.

        Args:
            cluster: Error cluster
            current_iteration: Current iteration

        Returns:
            True if this is a gap needing a new agent
        """
        # Must have enough errors
        if cluster.error_count < self.error_cluster_min_size:
            return False

        # Must persist long enough
        if cluster.persistence(current_iteration) < self.persistence_threshold:
            return False

        # Must have high average magnitude (not well handled)
        if cluster.avg_magnitude < 0.5:
            return False

        return True

    def find_gaps(self, current_iteration: int) -> List[ErrorCluster]:
        """Find all knowledge gaps."""
        gaps = []

        for cluster in self.error_clusters.values():
            if self.is_knowledge_gap(cluster, current_iteration):
                gaps.append(cluster)

        # Sort by magnitude (most problematic first)
        gaps.sort(key=lambda c: c.avg_magnitude, reverse=True)

        return gaps

    def create_agent_for_gap(
        self,
        cluster: ErrorCluster,
        current_iteration: int
    ) -> Optional[Agent]:
        """
        Create a new agent for a knowledge gap.

        Args:
            cluster: Error cluster representing the gap
            current_iteration: Current iteration

        Returns:
            New agent or None if at capacity
        """
        # Check capacity
        if self.hierarchy.count_agents() >= self.max_agents:
            return None

        # Find best parent
        parent = self._find_best_parent(cluster)

        if parent is None:
            parent = self.hierarchy.root

        # Create new agent
        new_agent = Agent(
            agent_id=f"grown_{current_iteration}_{cluster.cluster_id}",
            agent_type='agent' if parent.agent_type in ['master', 'manager'] else 'sub_agent',
            specialty=f"ErrorPattern_{cluster.cluster_id}",
            input_dim=parent.input_dim,
            hidden_dim=parent.hidden_dim,
            output_dim=parent.output_dim,
            initial_trust=self.new_agent_trust,
            creation_iteration=current_iteration
        )

        # Initialize weights biased toward error cluster
        # (slight nudge in error direction)
        if cluster.centroid is not None and len(cluster.centroid) == new_agent.output_dim:
            bias_factor = 0.1
            new_agent.weights['b2'] += bias_factor * cluster.centroid

        # Add to hierarchy
        self.hierarchy.add_agent(new_agent, parent)

        # Add to validation queue
        self.validating_agents.append({
            'agent': new_agent,
            'start_iteration': current_iteration,
            'cluster_id': cluster.cluster_id,
            'initial_magnitude': cluster.avg_magnitude
        })

        # Record
        self.creation_history.append({
            'iteration': current_iteration,
            'agent_id': new_agent.id,
            'cluster_id': cluster.cluster_id,
            'parent_id': parent.id
        })
        self.total_created += 1

        return new_agent

    def _find_best_parent(self, cluster: ErrorCluster) -> Optional[Agent]:
        """Find best parent for a new agent based on cluster characteristics."""
        # Simple heuristic: find manager with lowest average trust
        # (assuming it might need help)
        managers = self.hierarchy.get_agents_by_type('manager')

        if not managers:
            return self.hierarchy.root

        # Return manager with fewest children (balance tree)
        return min(managers, key=lambda m: len(m.children))

    def validate_new_agents(self, current_iteration: int) -> dict:
        """
        Validate agents in validation period.

        Args:
            current_iteration: Current iteration

        Returns:
            Validation results
        """
        validated = []
        removed = []
        still_validating = []

        for entry in list(self.validating_agents):
            agent = entry['agent']
            start = entry['start_iteration']
            cluster_id = entry['cluster_id']
            initial_mag = entry['initial_magnitude']

            # Check if validation period complete
            if current_iteration - start < self.validation_period:
                still_validating.append(agent.id)
                continue

            # Check if cluster magnitude improved
            cluster = self.error_clusters.get(cluster_id)

            if cluster and cluster.avg_magnitude < initial_mag * 0.8:
                # Improved by at least 20% - keep agent
                agent.trust = 0.5  # Boost trust
                validated.append(agent.id)
            elif agent.trust < 0.2:
                # Failed to help and low trust - remove
                self.hierarchy.remove_agent(agent)
                removed.append(agent.id)
            else:
                # Marginal - keep but don't boost
                validated.append(agent.id)

            self.validating_agents.remove(entry)

        return {
            'validated': validated,
            'removed': removed,
            'still_validating': still_validating
        }

    def execute_growing_cycle(self, current_iteration: int, max_new: int = 2) -> dict:
        """
        Execute growing cycle.

        Args:
            current_iteration: Current iteration
            max_new: Maximum new agents to create

        Returns:
            Statistics
        """
        # Find gaps
        gaps = self.find_gaps(current_iteration)

        created = []

        for gap in gaps[:max_new]:
            new_agent = self.create_agent_for_gap(gap, current_iteration)
            if new_agent:
                created.append(new_agent.id)

                # Reset cluster to track if new agent helps
                gap.error_count = 0
                gap.magnitudes = []

        # Validate existing new agents
        validation_results = self.validate_new_agents(current_iteration)

        return {
            'gaps_found': len(gaps),
            'agents_created': len(created),
            'created_ids': created,
            'validation': validation_results
        }

    def update_thresholds(
        self,
        min_size: int = None,
        persistence: int = None,
        max_agents: int = None
    ):
        """Update growing thresholds."""
        if min_size is not None:
            self.error_cluster_min_size = max(20, min(500, min_size))
        if persistence is not None:
            self.persistence_threshold = max(1000, min(20000, persistence))
        if max_agents is not None:
            self.max_agents = max(10, min(500, max_agents))

    def get_statistics(self) -> dict:
        """Get growing statistics."""
        return {
            'total_created': self.total_created,
            'under_validation': len(self.validating_agents),
            'active_clusters': len(self.error_clusters),
            'error_history_size': len(self.error_history)
        }
