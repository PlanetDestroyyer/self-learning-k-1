"""
Hierarchy management for K-1 Self-Learning System.

Manages the tree structure of agents.
"""

import numpy as np
from typing import List, Dict, Optional, Set
from .agent import Agent


class Hierarchy:
    """
    Manages the hierarchical structure of agents.
    """

    def __init__(self, max_depth: int = 6):
        """
        Initialize hierarchy.

        Args:
            max_depth: Maximum depth of hierarchy tree
        """
        self.root: Optional[Agent] = None
        self.max_depth = max_depth
        self.all_agents: Dict[str, Agent] = {}

    def set_root(self, agent: Agent):
        """Set the root agent (Master Manager)."""
        self.root = agent
        self._register_agent(agent)

    def _register_agent(self, agent: Agent):
        """Register an agent in the hierarchy."""
        self.all_agents[agent.id] = agent

    def _unregister_agent(self, agent: Agent):
        """Unregister an agent from the hierarchy."""
        if agent.id in self.all_agents:
            del self.all_agents[agent.id]

    def add_agent(self, agent: Agent, parent: Agent):
        """
        Add an agent under a parent.

        Args:
            agent: Agent to add
            parent: Parent agent
        """
        if self.get_depth(parent) >= self.max_depth:
            return False

        parent.add_child(agent)
        self._register_agent(agent)
        return True

    def remove_agent(self, agent: Agent):
        """
        Remove an agent from hierarchy.

        Args:
            agent: Agent to remove
        """
        if agent == self.root:
            return False

        # Reassign children to parent
        if agent.parent:
            for child in list(agent.children):
                agent.remove_child(child)
                agent.parent.add_child(child)

            agent.parent.remove_child(agent)

        self._unregister_agent(agent)
        return True

    def move_agent(self, agent: Agent, new_parent: Agent) -> bool:
        """
        Move an agent to a new parent.

        Args:
            agent: Agent to move
            new_parent: New parent agent

        Returns:
            True if successful
        """
        if agent == self.root:
            return False

        if self._would_create_cycle(agent, new_parent):
            return False

        if self.get_depth(new_parent) >= self.max_depth - 1:
            return False

        # Remove from current parent
        if agent.parent:
            agent.parent.remove_child(agent)

        # Add to new parent
        new_parent.add_child(agent)

        return True

    def _would_create_cycle(self, agent: Agent, potential_parent: Agent) -> bool:
        """Check if making potential_parent the parent would create a cycle."""
        current = potential_parent
        while current is not None:
            if current == agent:
                return True
            current = current.parent
        return False

    def get_depth(self, agent: Agent) -> int:
        """Get depth of an agent in the tree."""
        depth = 0
        current = agent
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def get_path_to_root(self, agent: Agent) -> List[Agent]:
        """Get path from agent to root."""
        path = []
        current = agent
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def find_agent(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID."""
        return self.all_agents.get(agent_id)

    def get_all_agents(self) -> List[Agent]:
        """Get list of all agents."""
        return list(self.all_agents.values())

    def count_agents(self) -> int:
        """Count total number of agents."""
        return len(self.all_agents)

    def count_by_type(self) -> Dict[str, int]:
        """Count agents by type."""
        counts = {}
        for agent in self.all_agents.values():
            counts[agent.agent_type] = counts.get(agent.agent_type, 0) + 1
        return counts

    def get_agents_by_type(self, agent_type: str) -> List[Agent]:
        """Get all agents of a specific type."""
        return [a for a in self.all_agents.values() if a.agent_type == agent_type]

    def get_siblings(self, agent: Agent) -> List[Agent]:
        """Get sibling agents (same parent)."""
        if agent.parent is None:
            return []
        return [a for a in agent.parent.children if a != agent]

    def find_closest_manager(self, specialty: str) -> Optional[Agent]:
        """Find manager closest to a specialty (simple keyword matching)."""
        managers = self.get_agents_by_type('manager')

        if not managers:
            return self.root

        specialty_lower = specialty.lower()
        best_match = None
        best_score = 0

        for manager in managers:
            # Simple word overlap scoring
            manager_words = set(manager.specialty.lower().split())
            specialty_words = set(specialty_lower.split())
            overlap = len(manager_words & specialty_words)

            if overlap > best_score:
                best_score = overlap
                best_match = manager

        return best_match if best_match else self.root

    def get_leaves(self) -> List[Agent]:
        """Get all leaf agents (no children)."""
        return [a for a in self.all_agents.values() if len(a.children) == 0]

    def get_managers(self) -> List[Agent]:
        """Get all non-leaf agents."""
        return [a for a in self.all_agents.values() if len(a.children) > 0]

    def print_tree(self, agent: Agent = None, indent: int = 0):
        """Print tree structure."""
        if agent is None:
            agent = self.root

        if agent is None:
            print("(empty hierarchy)")
            return

        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        print(f"{prefix}{agent.id} ({agent.agent_type}, trust={agent.trust:.3f})")

        for child in agent.get_children_ordered():
            self.print_tree(child, indent + 1)

    def get_statistics(self) -> Dict:
        """Get hierarchy statistics."""
        agents = self.get_all_agents()

        if not agents:
            return {'total': 0}

        trusts = [a.trust for a in agents]
        depths = [self.get_depth(a) for a in agents]

        return {
            'total': len(agents),
            'by_type': self.count_by_type(),
            'avg_trust': float(np.mean(trusts)),
            'trust_std': float(np.std(trusts)),
            'max_depth': max(depths),
            'num_leaves': len(self.get_leaves()),
            'num_managers': len(self.get_managers())
        }


def build_initial_hierarchy(
    input_dim: int = 128,
    hidden_dim: int = 128,
    output_dim: int = 128,
    initial_trust: float = 0.3,
    domains: List[tuple] = None
) -> Hierarchy:
    """
    Build initial hierarchical structure.

    Args:
        input_dim: Input dimension for agents
        hidden_dim: Hidden dimension for agents
        output_dim: Output dimension for agents
        initial_trust: Initial trust score
        domains: List of (domain_name, num_agents) tuples

    Returns:
        Initialized Hierarchy
    """
    if domains is None:
        domains = [
            ('Syntax', 5),
            ('Semantics', 5),
            ('Vocabulary', 5),
            ('Context', 5),
        ]

    hierarchy = Hierarchy(max_depth=4)

    # Create root (Master Manager)
    root = Agent(
        agent_id='master',
        agent_type='master',
        specialty='Language Model',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        initial_trust=initial_trust,
        creation_iteration=0
    )
    hierarchy.set_root(root)

    # Create domain managers and their agents
    for domain_name, num_agents in domains:
        # Create manager
        manager = Agent(
            agent_id=f'mgr_{domain_name.lower()}',
            agent_type='manager',
            specialty=domain_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            initial_trust=initial_trust,
            creation_iteration=0
        )
        hierarchy.add_agent(manager, root)

        # Create agents under manager
        for i in range(num_agents):
            agent = Agent(
                agent_id=f'agent_{domain_name.lower()}_{i}',
                agent_type='agent',
                specialty=f'{domain_name}_{i}',
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                initial_trust=initial_trust,
                creation_iteration=0
            )
            hierarchy.add_agent(agent, manager)

    return hierarchy
