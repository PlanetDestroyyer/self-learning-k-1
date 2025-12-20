"""
Hierarchy builder for initializing the agent structure.

Builds the initial hierarchical structure based on domain analysis.
"""

import torch
from typing import Dict, List
from ..core.agent import Agent
from ..core.hierarchy import Hierarchy
from .domain_analyzer import DomainAnalyzer


class HierarchyBuilder:
    """
    Builds initial hierarchical structure from domain analysis.
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 64,
                 output_dim: int = 128,
                 initial_trust: float = 0.3):
        """
        Initialize hierarchy builder.

        Args:
            input_dim: Input dimension for agents
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            initial_trust: Initial trust for all agents
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.initial_trust = initial_trust

        # Device handling for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_hierarchy(self,
                       domain_analysis: Dict,
                       max_depth: int = 6) -> Hierarchy:
        """
        Build hierarchy from domain analysis.

        Args:
            domain_analysis: Results from DomainAnalyzer
            max_depth: Maximum hierarchy depth

        Returns:
            Initialized Hierarchy
        """
        hierarchy = Hierarchy(max_depth=max_depth)

        # Create root (Master Manager)
        root = Agent(
            agent_id='master_manager',
            agent_type='master',
            specialty='Master Manager',
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            initial_trust=self.initial_trust,
            creation_iteration=0
        ).to(self.device)

        hierarchy.set_root(root)

        # Build from hierarchy structure
        domain_hierarchy = domain_analysis.get('hierarchy', {})
        domains = domain_analysis.get('domains', {})

        # Create managers for top-level domains
        for manager_name, children_names in domain_hierarchy.items():
            manager = self._create_agent(
                agent_id=f"manager_{manager_name.lower().replace(' ', '_')}",
                agent_type='manager',
                specialty=manager_name
            )

            hierarchy.add_agent(manager, root)

            # Create agents for each child domain
            for child_name in children_names:
                agent = self._create_agent(
                    agent_id=f"agent_{child_name.lower().replace(' ', '_')}",
                    agent_type='agent',
                    specialty=child_name
                )

                hierarchy.add_agent(agent, manager)

                # Create sub-agents for further specialization
                sub_agents = self._create_sub_agents(child_name)
                for sub_agent in sub_agents:
                    hierarchy.add_agent(sub_agent, agent)

        # For domains without explicit hierarchy, create directly under root
        for domain_name, domain_info in domains.items():
            # Skip if already added through hierarchy
            if any(domain_name in children for children in domain_hierarchy.values()):
                continue

            agent = self._create_agent(
                agent_id=f"agent_{domain_name.lower().replace(' ', '_')}",
                agent_type='agent',
                specialty=domain_name
            )

            hierarchy.add_agent(agent, root)

        return hierarchy

    def build_simple_hierarchy(self,
                              num_domains: int = 5,
                              agents_per_domain: int = 3) -> Hierarchy:
        """
        Build a simple hierarchy for testing.

        Args:
            num_domains: Number of domain managers
            agents_per_domain: Number of agents per domain

        Returns:
            Simple hierarchy
        """
        hierarchy = Hierarchy(max_depth=4)

        # Create root
        root = Agent(
            agent_id='master_manager',
            agent_type='master',
            specialty='Master Manager',
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            initial_trust=self.initial_trust,
            creation_iteration=0
        ).to(self.device)

        hierarchy.set_root(root)

        # Create domains
        for domain_idx in range(num_domains):
            manager = self._create_agent(
                agent_id=f"manager_{domain_idx}",
                agent_type='manager',
                specialty=f"Domain_{domain_idx}"
            )

            hierarchy.add_agent(manager, root)

            # Create agents under each domain
            for agent_idx in range(agents_per_domain):
                agent = self._create_agent(
                    agent_id=f"agent_{domain_idx}_{agent_idx}",
                    agent_type='agent',
                    specialty=f"Specialist_{domain_idx}_{agent_idx}"
                )

                hierarchy.add_agent(agent, manager)

        return hierarchy

    def build_from_dataset(self,
                          dataset_type: str = 'general',
                          num_classes: int = 10) -> Hierarchy:
        """
        Build hierarchy for a specific dataset type.

        Args:
            dataset_type: Type of dataset
            num_classes: Number of classes (for classification)

        Returns:
            Hierarchy tailored to dataset
        """
        analyzer = DomainAnalyzer()
        domain_structure = analyzer.get_predefined_structure(dataset_type)

        return self.build_hierarchy(domain_structure)

    def _create_agent(self,
                     agent_id: str,
                     agent_type: str,
                     specialty: str) -> Agent:
        """
        Create an agent with default parameters and move to device.

        Args:
            agent_id: Agent ID
            agent_type: Agent type
            specialty: Agent specialty

        Returns:
            Created agent (on correct device)
        """
        agent = Agent(
            agent_id=agent_id,
            agent_type=agent_type,
            specialty=specialty,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            initial_trust=self.initial_trust,
            creation_iteration=0
        )
        # Move to device (GPU/CPU)
        return agent.to(self.device)

    def _create_sub_agents(self, parent_specialty: str, num_sub_agents: int = 2) -> List[Agent]:
        """
        Create sub-agents for further specialization.

        Args:
            parent_specialty: Parent specialty
            num_sub_agents: Number of sub-agents to create

        Returns:
            List of sub-agents
        """
        sub_agents = []

        for i in range(num_sub_agents):
            sub_agent = self._create_agent(
                agent_id=f"sub_{parent_specialty.lower().replace(' ', '_')}_{i}",
                agent_type='sub_agent',
                specialty=f"{parent_specialty} Specialist {i+1}"
            )
            sub_agents.append(sub_agent)

        return sub_agents

    def print_structure(self, hierarchy: Hierarchy):
        """
        Print the created hierarchy structure.

        Args:
            hierarchy: Hierarchy to print
        """
        print("\n" + "=" * 60)
        print("INITIAL HIERARCHY STRUCTURE")
        print("=" * 60)
        hierarchy.print_tree()
        print("=" * 60)
        print(f"Total agents: {hierarchy.count_agents()}")
        print(f"By type: {hierarchy.count_agents_by_type()}")
        print("=" * 60 + "\n")
