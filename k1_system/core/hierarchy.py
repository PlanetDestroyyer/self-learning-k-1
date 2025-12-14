"""
Hierarchy management for the Self-Learning K-1 System.

Manages the tree structure of agents, including traversal,
finding agents, and structural modifications.
"""

from typing import List, Optional, Set, Dict, Tuple
from collections import deque
import numpy as np
from .agent import Agent


class Hierarchy:
    """
    Manages the hierarchical structure of agents.

    The hierarchy is a tree structure with:
    - Level 0: Master Manager (root)
    - Level 1: Domain Managers
    - Level 2: Agents
    - Level 3: Sub-agents
    """

    def __init__(self, max_depth: int = 6):
        """
        Initialize hierarchy.

        Args:
            max_depth: Maximum depth of hierarchy
        """
        self.root: Optional[Agent] = None
        self.max_depth = max_depth
        self._all_agents: Dict[str, Agent] = {}
        self._agents_by_type: Dict[str, List[Agent]] = {
            'master': [],
            'manager': [],
            'agent': [],
            'sub_agent': []
        }

    def set_root(self, root: Agent):
        """Set the root (Master Manager) of the hierarchy."""
        self.root = root
        self._register_agent(root)

    def _register_agent(self, agent: Agent):
        """Register an agent in internal tracking."""
        self._all_agents[agent.id] = agent
        if agent.agent_type in self._agents_by_type:
            if agent not in self._agents_by_type[agent.agent_type]:
                self._agents_by_type[agent.agent_type].append(agent)

    def _unregister_agent(self, agent: Agent):
        """Unregister an agent from internal tracking."""
        if agent.id in self._all_agents:
            del self._all_agents[agent.id]
        if agent.agent_type in self._agents_by_type:
            if agent in self._agents_by_type[agent.agent_type]:
                self._agents_by_type[agent.agent_type].remove(agent)

    def add_agent(self, agent: Agent, parent: Agent):
        """
        Add an agent to the hierarchy under a parent.

        Args:
            agent: Agent to add
            parent: Parent agent
        """
        parent.add_child(agent)
        self._register_agent(agent)

    def remove_agent(self, agent: Agent):
        """
        Remove an agent from the hierarchy.

        Args:
            agent: Agent to remove
        """
        # First remove all children
        children_copy = list(agent.children)
        for child in children_copy:
            self.remove_agent(child)

        # Remove from parent
        if agent.parent is not None:
            agent.parent.remove_child(agent)

        # Unregister
        self._unregister_agent(agent)

    def find_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Find an agent by ID.

        Args:
            agent_id: Agent ID to find

        Returns:
            Agent if found, None otherwise
        """
        return self._all_agents.get(agent_id)

    def get_all_agents(self) -> List[Agent]:
        """Get all agents in the hierarchy."""
        return list(self._all_agents.values())

    def get_agents_by_type(self, agent_type: str) -> List[Agent]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Type of agent

        Returns:
            List of agents of that type
        """
        return self._agents_by_type.get(agent_type, [])

    def get_depth(self, agent: Agent) -> int:
        """
        Get depth of an agent in the hierarchy.

        Args:
            agent: Agent to check

        Returns:
            Depth (0 for root)
        """
        depth = 0
        current = agent
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def get_path_to_root(self, agent: Agent) -> List[Agent]:
        """
        Get path from agent to root.

        Args:
            agent: Starting agent

        Returns:
            List of agents from agent to root
        """
        path = []
        current = agent
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def get_all_descendants(self, agent: Agent) -> List[Agent]:
        """
        Get all descendants of an agent (BFS).

        Args:
            agent: Parent agent

        Returns:
            List of all descendants
        """
        descendants = []
        queue = deque(list(agent.children))

        while queue:
            current = queue.popleft()
            descendants.append(current)
            queue.extend(current.children)

        return descendants

    def get_siblings(self, agent: Agent) -> List[Agent]:
        """
        Get siblings of an agent.

        Args:
            agent: Agent to find siblings for

        Returns:
            List of sibling agents
        """
        if agent.parent is None:
            return []

        return [child for child in agent.parent.children if child != agent]

    def count_agents(self) -> int:
        """Count total number of agents."""
        return len(self._all_agents)

    def count_agents_by_type(self) -> Dict[str, int]:
        """Count agents by type."""
        return {
            agent_type: len(agents)
            for agent_type, agents in self._agents_by_type.items()
        }

    def find_agents_by_specialty(self, specialty: str) -> List[Agent]:
        """
        Find agents by specialty.

        Args:
            specialty: Specialty to search for

        Returns:
            List of agents with that specialty
        """
        return [agent for agent in self._all_agents.values()
                if specialty.lower() in agent.specialty.lower()]

    def find_closest_manager(self, specialty: str) -> Agent:
        """
        Find the closest manager for a given specialty.

        Args:
            specialty: Specialty to match

        Returns:
            Best matching manager or root
        """
        managers = self.get_agents_by_type('manager')

        if not managers:
            return self.root

        # Simple similarity based on specialty string overlap
        best_manager = managers[0]
        best_score = 0

        for manager in managers:
            # Count common words
            spec_words = set(specialty.lower().split())
            manager_words = set(manager.specialty.lower().split())
            overlap = len(spec_words & manager_words)

            if overlap > best_score:
                best_score = overlap
                best_manager = manager

        return best_manager if best_score > 0 else self.root

    def move_agent(self, agent: Agent, new_parent: Agent) -> bool:
        """
        Move an agent to a new parent.

        Args:
            agent: Agent to move
            new_parent: New parent

        Returns:
            True if successful
        """
        # Check if move would create cycle
        if self._would_create_cycle(agent, new_parent):
            return False

        # Check depth constraints
        if self.get_depth(new_parent) + 1 >= self.max_depth:
            return False

        # Remove from old parent
        if agent.parent is not None:
            agent.parent.remove_child(agent)

        # Add to new parent
        new_parent.add_child(agent)

        return True

    def _would_create_cycle(self, agent: Agent, new_parent: Agent) -> bool:
        """Check if moving agent under new_parent would create a cycle."""
        current = new_parent
        while current is not None:
            if current == agent:
                return True
            current = current.parent
        return False

    def replace_agent(self, old_agent: Agent, new_agent: Agent):
        """
        Replace an agent with a new one, maintaining children.

        Args:
            old_agent: Agent to replace
            new_agent: New agent
        """
        # Transfer children
        children_copy = list(old_agent.children)
        for child in children_copy:
            old_agent.remove_child(child)
            new_agent.add_child(child)

        # Set same parent
        parent = old_agent.parent
        if parent is not None:
            parent.remove_child(old_agent)
            parent.add_child(new_agent)

        # Register new, unregister old
        self._unregister_agent(old_agent)
        self._register_agent(new_agent)

        # If old agent was root, update root
        if old_agent == self.root:
            self.root = new_agent

    def get_leaves(self) -> List[Agent]:
        """Get all leaf agents (no children)."""
        return [agent for agent in self._all_agents.values()
                if len(agent.children) == 0]

    def get_active_agents(self, threshold: float = 0.01) -> List[Agent]:
        """
        Get agents with recent activation above threshold.

        Args:
            threshold: Minimum average activation

        Returns:
            List of active agents
        """
        return [agent for agent in self._all_agents.values()
                if agent.get_avg_activation() > threshold]

    def compute_activation_rate(self, agent: Agent) -> float:
        """
        Compute activation rate for an agent.

        Args:
            agent: Agent to check

        Returns:
            Activation rate (0.0 to 1.0)
        """
        return agent.get_avg_activation()

    def get_statistics(self) -> Dict:
        """
        Get hierarchy statistics.

        Returns:
            Dictionary of statistics
        """
        all_agents = self.get_all_agents()

        if not all_agents:
            return {}

        trust_scores = [agent.trust for agent in all_agents]
        activation_rates = [agent.get_avg_activation() for agent in all_agents]

        return {
            'total_agents': len(all_agents),
            'by_type': self.count_agents_by_type(),
            'avg_trust': np.mean(trust_scores),
            'std_trust': np.std(trust_scores),
            'min_trust': np.min(trust_scores),
            'max_trust': np.max(trust_scores),
            'avg_activation': np.mean(activation_rates),
            'active_agents': len([a for a in all_agents if a.get_avg_activation() > 0.01]),
            'leaf_agents': len(self.get_leaves()),
            'max_depth': max([self.get_depth(a) for a in all_agents]) if all_agents else 0
        }

    def print_tree(self, agent: Optional[Agent] = None, prefix: str = "", is_last: bool = True):
        """
        Print the hierarchy tree structure.

        Args:
            agent: Starting agent (root if None)
            prefix: Prefix for formatting
            is_last: Whether this is the last child
        """
        if agent is None:
            agent = self.root

        if agent is None:
            print("Empty hierarchy")
            return

        # Print current agent
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{agent.specialty} (trust={agent.trust:.3f}, "
              f"children={len(agent.children)})")

        # Print children
        children = list(agent.children)
        for i, child in enumerate(children):
            extension = "    " if is_last else "│   "
            self.print_tree(child, prefix + extension, i == len(children) - 1)

    def validate_structure(self) -> List[str]:
        """
        Validate hierarchy structure and return list of issues.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check root exists
        if self.root is None:
            issues.append("No root agent")
            return issues

        # Check all agents are reachable from root
        reachable = set([self.root.id])
        queue = deque([self.root])
        while queue:
            current = queue.popleft()
            for child in current.children:
                reachable.add(child.id)
                queue.append(child)

        for agent_id in self._all_agents:
            if agent_id not in reachable:
                issues.append(f"Agent {agent_id} not reachable from root")

        # Check for cycles
        visited = set()
        def check_cycle(agent, path):
            if agent.id in path:
                issues.append(f"Cycle detected involving {agent.id}")
                return
            path.add(agent.id)
            visited.add(agent.id)
            for child in agent.children:
                check_cycle(child, path.copy())

        check_cycle(self.root, set())

        # Check depth constraints
        for agent in self._all_agents.values():
            depth = self.get_depth(agent)
            if depth > self.max_depth:
                issues.append(f"Agent {agent.id} exceeds max depth ({depth} > {self.max_depth})")

        return issues
