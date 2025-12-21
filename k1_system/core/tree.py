"""
HierarchicalTree: The main K-1 tree structure for language modeling.

Architecture:
                    ROOT (Manager)
                         |
         ┌───────────────┼───────────────┐
         |               |               |
      Node 1          Node 2          Node 3
         |               |               |
    ┌────┼────┐     ┌────┼────┐    ┌────┼────┐
   L1   L2   L3    L4   L5   L6   L7   L8   L9

Data flows DOWN through the tree, gradients flow UP.
Only the PATH responsible for errors gets updated.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .tree_node import TreeNode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HierarchicalTree(nn.Module):
    """
    Hierarchical tree of nodes for language modeling.
    
    Structure:
    - Shared embedding layer
    - Tree of TreeNode modules (depth configurable)
    - Shared output projection
    
    Key innovation: Path-based gradient updates with improved responsibility signal.
    
    Attributes:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        tree_depth: Number of levels in tree
        branching_factor: Children per node at each level
        all_nodes: List of all TreeNode instances
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        ff_dim: int = 256,
        num_heads: int = 4,
        tree_depth: int = 4,          # Root + Nodes + Agents + Sub-Agents
        branching_factor=None,        # [4, 3, 2] or single int
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize HierarchicalTree.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            ff_dim: Feed-forward hidden dimension
            num_heads: Number of attention heads
            tree_depth: Depth of tree (e.g., 4 = Root→Nodes→Agents→SubAgents)
            branching_factor: Children per node, e.g. [4, 3, 2]
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.tree_depth = tree_depth
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Support variable branching: [4 nodes, 3 agents, 2 sub-agents]
        if branching_factor is None:
            branching_factor = [4, 3, 2]
        elif isinstance(branching_factor, int):
            branching_factor = [branching_factor] * (tree_depth - 1)

        self.branching_factor = branching_factor
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)
        
        # Build hierarchical tree
        self.root = self._build_tree(depth=0, max_depth=tree_depth, node_id=0)
        
        # Collect all nodes for easy access
        self.all_nodes: List[TreeNode] = []
        self._collect_nodes(self.root)
        
        # Shared output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Print structure
        print(f"Created HierarchicalTree:")
        print(f"  Vocab: {vocab_size}, Embed: {embed_dim}")
        print(f"  Tree depth: {tree_depth}, Branching: {branching_factor}")
        print(f"  Total nodes: {len(self.all_nodes)}")
        print(f"  Leaf nodes: {sum(1 for n in self.all_nodes if n.is_leaf)}")
    
    def _build_tree(self, depth: int, max_depth: int, node_id: int) -> TreeNode:
        """
        Recursively build the tree with variable branching.

        Structure:
          Root (depth=0, hidden)
            → 4 Nodes (depth=1)
              → 3 Agents per Node (depth=2)
                → 2 Sub-Agents per Agent (depth=3)
        """
        node = TreeNode(self.embed_dim, self.ff_dim, self.num_heads, self.dropout)
        node.node_id = node_id
        node.level = depth

        if depth < max_depth - 1:
            # Get branching factor for this level
            if isinstance(self.branching_factor, list):
                num_children = (
                    self.branching_factor[depth] 
                    if depth < len(self.branching_factor) 
                    else self.branching_factor[-1]
                )
            else:
                num_children = self.branching_factor

            # Create children
            for i in range(num_children):
                child_id = (
                    len(self.all_nodes) 
                    if hasattr(self, 'all_nodes') 
                    else node_id * 10 + i + 1
                )
                child = self._build_tree(depth + 1, max_depth, child_id)
                node.add_child(child)

        return node
    
    def _collect_nodes(self, node: TreeNode):
        """Collect all nodes into a list."""
        self.all_nodes.append(node)
        for child in node.child_nodes:
            self._collect_nodes(child)
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[TreeNode]]:
        """
        Forward pass through the tree.
        
        Args:
            x: Input token indices [batch, seq_len]
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            path: List of nodes that processed the input
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens
        h = self.embedding(x)  # [batch, seq, embed]
        h = h + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Process through tree (all nodes)
        path = []
        h = self._forward_node(h, self.root, mask, path)
        
        # Output projection
        h = self.output_norm(h)
        logits = self.output_proj(h)
        
        return logits, path
    
    def _forward_node(
        self, 
        x: torch.Tensor, 
        node: TreeNode, 
        mask: torch.Tensor, 
        path: List[TreeNode]
    ) -> torch.Tensor:
        """Recursively forward through a node and its children."""
        # Process through this node
        x = node(x, mask)
        path.append(node)

        # If has children, process through them (combine outputs)
        if node.child_nodes:
            child_outputs = []
            for child in node.child_nodes:
                child_out = self._forward_node(x, child, mask, path)
                child_outputs.append(child_out)

            # Average child outputs (can be changed to routing later)
            x = torch.stack(child_outputs, dim=0).mean(dim=0)

        return x
    
    # ========================================
    # Responsibility Signal Methods
    # ========================================
    
    def _get_node_gradient_norm(self, node: TreeNode) -> float:
        """Get gradient norm for a specific node."""
        total = 0.0
        for p in node.parameters():
            if p.grad is not None:
                total += p.grad.norm().item()
        return total

    def compute_responsibility(
        self, 
        node: TreeNode, 
        loss: float, 
        current_step: int,
        cooldown_steps: int = 50, 
        penalty: float = 0.1
    ) -> float:
        """
        Compute responsibility score for a node.
        
        responsibility = gradient × loss × cooldown_penalty
        
        Args:
            node: The node to compute responsibility for
            loss: Current batch loss value
            current_step: Current training step
            cooldown_steps: Steps before cooldown expires
            penalty: Minimum multiplier for recently-updated nodes
        
        Returns:
            Responsibility score (higher = more responsible for error)
        """
        grad = self._get_node_gradient_norm(node)
        
        # Cooldown penalty: recently updated nodes get lower responsibility
        steps_since_update = current_step - node.last_updated_step
        if steps_since_update < cooldown_steps:
            cooldown_factor = penalty + (1 - penalty) * (steps_since_update / cooldown_steps)
        else:
            cooldown_factor = 1.0
        
        return grad * loss * cooldown_factor

    def find_responsible_path(
        self, 
        loss: float = 1.0, 
        current_step: int = 0
    ) -> List[Tuple[TreeNode, float]]:
        """
        Hierarchically drill down from Manager to find responsible agent path.
        
        Uses improved responsibility signal: gradient × loss × cooldown_penalty

        Args:
            loss: Current batch loss value (higher = weight responsibility more)
            current_step: Current training step (for cooldown calculation)

        Returns:
            List of (node, responsibility_score) from Manager → Agent → Sub-Agent
        """
        path = []
        current = self.root

        while not current.is_leaf:
            resp = self.compute_responsibility(current, loss, current_step)
            path.append((current, resp))

            # Find child with highest responsibility
            if current.child_nodes:
                child_resps = [
                    (child, self.compute_responsibility(child, loss, current_step))
                    for child in current.child_nodes
                ]
                culprit_child, _ = max(child_resps, key=lambda x: x[1])
                current = culprit_child

        # Add leaf (sub-agent culprit)
        path.append((current, self.compute_responsibility(current, loss, current_step)))
        return path

    def mark_nodes_updated(self, scales: dict, current_step: int):
        """
        Mark nodes that received updates with current step.
        
        This enables the trust cooldown mechanism.
        
        Args:
            scales: Dict of node_id -> update_scale
            current_step: Current training step
        """
        for node in self.all_nodes:
            if scales.get(node.node_id, 0.0) > 0:
                node.last_updated_step = current_step

    def get_proportional_scales(
        self, 
        responsible_path: List[Tuple[TreeNode, float]]
    ) -> dict:
        """
        Compute proportional update scales based on hierarchical responsibility.

        Args:
            responsible_path: Path from find_responsible_path()

        Returns:
            Dict mapping node_id to update scale (0.0 to 1.0)

        Example:
            Sub-Agent (culprit): 1.0 (100% update)
            Agent (parent):      0.15 (15% update)
            Manager (root):      0.05 (5% update)
            Others:              0.0 (skip)
        """
        scales = {}
        path_nodes = [node for node, _ in responsible_path]

        for node in self.all_nodes:
            if node == path_nodes[-1]:  # Culprit (deepest in path)
                scales[node.node_id] = 1.0
            elif len(path_nodes) > 1 and node == path_nodes[-2]:  # Parent
                scales[node.node_id] = 0.15
            elif node == path_nodes[0]:  # Root
                scales[node.node_id] = 0.05
            else:
                scales[node.node_id] = 0.0

        return scales

    def apply_proportional_updates(self, scales: dict):
        """
        Scale gradients proportionally based on hierarchical responsibility.

        Args:
            scales: Dict from get_proportional_scales()
        """
        for node in self.all_nodes:
            scale = scales.get(node.node_id, 0.0)
            for p in node.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

    # ========================================
    # Utility Methods
    # ========================================

    def get_path_gradient_norms(self) -> dict:
        """Get gradient norms for all nodes."""
        return {node.node_id: node.get_gradient_norm() for node in self.all_nodes}
    
    def get_node_by_id(self, node_id: int) -> Optional[TreeNode]:
        """Get node by ID."""
        for node in self.all_nodes:
            if node.node_id == node_id:
                return node
        return None

    def print_responsibility_tree(
        self, 
        grad_norms: dict, 
        scales: dict, 
        node: TreeNode = None, 
        level: int = 0
    ):
        """Print hierarchical responsibility visualization."""
        if node is None:
            node = self.root

        indent = "  " * level
        node_id = node.node_id
        grad = grad_norms.get(node_id, 0.0)
        scale = scales.get(node_id, 0.0)

        # Status icon
        if scale >= 0.8:
            icon = "🚨"  # Culprit
        elif scale > 0:
            icon = "⚠️"   # Parent/Manager
        else:
            icon = "✓"

        # Role name
        roles = {0: "Root   ", 1: "Node   ", 2: "Agent  "}
        role = roles.get(level, "SubAgent")

        print(f"{indent}{icon} {role} {node_id}: grad={grad:.3f}, update={scale*100:3.0f}%")

        for child in node.child_nodes:
            self.print_responsibility_tree(grad_norms, scales, child, level + 1)
