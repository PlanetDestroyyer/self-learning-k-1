"""
Snapshot and rollback system for K-1 Self-Learning System.
"""

import numpy as np
import pickle
from typing import Dict, List, Optional
from pathlib import Path
from copy import deepcopy


class Snapshot:
    """A snapshot of system state."""

    def __init__(self, iteration: int):
        self.iteration = iteration
        self.agent_weights: Dict[str, Dict] = {}
        self.agent_trusts: Dict[str, float] = {}
        self.hierarchy_structure: Dict = {}
        self.metrics: Dict = {}

    def save_agent(self, agent_id: str, weights: Dict, trust: float):
        """Save an agent's state."""
        self.agent_weights[agent_id] = {k: v.copy() for k, v in weights.items()}
        self.agent_trusts[agent_id] = trust


class SnapshotManager:
    """
    Manages system snapshots for rollback.
    """

    def __init__(
        self,
        max_snapshots: int = 5,
        snapshot_dir: str = 'snapshots'
    ):
        """
        Initialize snapshot manager.

        Args:
            max_snapshots: Maximum snapshots to keep
            snapshot_dir: Directory for snapshot files
        """
        self.max_snapshots = max_snapshots
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)

        self.snapshots: List[Snapshot] = []
        self.rollback_count = 0

    def create_snapshot(self, iteration: int, hierarchy, metrics: Dict = None) -> Snapshot:
        """
        Create a snapshot of current state.

        Args:
            iteration: Current iteration
            hierarchy: Hierarchy to snapshot
            metrics: Current metrics

        Returns:
            Created snapshot
        """
        snapshot = Snapshot(iteration)

        # Save all agent states
        for agent in hierarchy.get_all_agents():
            snapshot.save_agent(agent.id, agent.weights, agent.trust)

        # Save metrics
        if metrics:
            snapshot.metrics = metrics.copy()

        # Add to list
        self.snapshots.append(snapshot)

        # Remove old snapshots if over limit
        while len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        return snapshot

    def restore_snapshot(self, snapshot: Snapshot, hierarchy) -> bool:
        """
        Restore system to a snapshot.

        Args:
            snapshot: Snapshot to restore
            hierarchy: Hierarchy to restore into

        Returns:
            True if successful
        """
        try:
            for agent in hierarchy.get_all_agents():
                if agent.id in snapshot.agent_weights:
                    # Restore weights
                    for key, value in snapshot.agent_weights[agent.id].items():
                        agent.weights[key] = value.copy()

                    # Restore trust
                    agent.trust = snapshot.agent_trusts[agent.id]

            self.rollback_count += 1
            return True

        except Exception as e:
            print(f"Rollback failed: {e}")
            return False

    def get_latest_snapshot(self) -> Optional[Snapshot]:
        """Get most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_snapshot_by_iteration(self, iteration: int) -> Optional[Snapshot]:
        """Get snapshot closest to an iteration."""
        if not self.snapshots:
            return None

        return min(self.snapshots, key=lambda s: abs(s.iteration - iteration))

    def save_to_disk(self, snapshot: Snapshot, name: str):
        """Save snapshot to disk."""
        path = self.snapshot_dir / f'{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(snapshot, f)

    def load_from_disk(self, name: str) -> Optional[Snapshot]:
        """Load snapshot from disk."""
        path = self.snapshot_dir / f'{name}.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def get_rollback_count(self) -> int:
        """Get total rollback count."""
        return self.rollback_count

    def clear_snapshots(self):
        """Clear all snapshots."""
        self.snapshots.clear()
