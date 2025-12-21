import torch
import torch.nn as nn
import math
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
        
        # Gradient norm cache (optimization)
        self._grad_norm_cache: dict = {}

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the tree using LEVEL-WISE batched processing.
        
        This is much faster than recursive because all nodes at same level
        are processed in parallel.
        
        NOTE: Only returns logits (tensor) for DataParallel compatibility.
        Use get_last_path() to access the path after forward.
        
        Args:
            x: Input token indices [batch, seq_len]
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens
        h = self.embedding(x)  # [batch, seq, embed]
        h = h + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create causal mask ONCE
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Track path (stored internally for DataParallel compatibility)
        self._last_path = []
        
        # Level 0: Root
        h = self.root(h, mask)
        self._last_path.append(self.root)
        
        # Process remaining levels in PARALLEL
        current_nodes = list(self.root.child_nodes)
        
        while current_nodes:
            # Process ALL nodes at this level at once
            level_outputs = []
            next_level_nodes = []
            
            for node in current_nodes:
                out = node(h, mask)
                level_outputs.append(out)
                self._last_path.append(node)
                next_level_nodes.extend(node.child_nodes)
            
            # Average outputs from this level
            if level_outputs:
                h = torch.stack(level_outputs, dim=0).mean(dim=0)
            
            current_nodes = next_level_nodes
        
        # Output projection
        h = self.output_norm(h)
        logits = self.output_proj(h)
        
        return logits
    
    def get_last_path(self) -> List[TreeNode]:
        """Get the path from the last forward pass."""
        return getattr(self, '_last_path', self.all_nodes)
    
    # ========================================
    # Responsibility Signal Methods (OPTIMIZED)
    # ========================================
    
    def cache_all_gradient_norms(self):
        """Cache gradient norms for all nodes (call once per step)."""
        self._grad_norm_cache = {}
        for node in self.all_nodes:
            total = 0.0
            for p in node.parameters():
                if p.grad is not None:
                    total += p.grad.norm().item()
            # NaN protection
            if math.isnan(total) or math.isinf(total):
                total = 0.0001
            self._grad_norm_cache[node.node_id] = total
    
    def fast_hierarchical_step(self, loss_tensor: torch.Tensor, current_step: int):
        """
        GPU-ACCELERATED hierarchical error attribution.
        
        Combines gradient norm computation, path finding, and scale application
        into minimal Python loops with maximum GPU tensor operations.
        
        Args:
            loss_tensor: The loss tensor (still on GPU, don't call .item())
            current_step: Current training step
        """
        # Compute all gradient norms as tensors (stay on GPU)
        grad_norms = []
        for node in self.all_nodes:
            node_grad = torch.tensor(0.0, device=loss_tensor.device)
            for p in node.parameters():
                if p.grad is not None:
                    node_grad = node_grad + p.grad.norm()
            grad_norms.append(node_grad)
        
        # Stack into single tensor (GPU)
        grad_tensor = torch.stack(grad_norms)  # [num_nodes]
        
        # Cache for logging (convert to dict only if needed later)
        self._grad_tensor = grad_tensor
        
        # Find responsible path using GPU operations
        # Build tree structure as indices for vectorized lookup
        # For now, use simple max-gradient path (3 nodes: root -> middle -> leaf)
        
        # Root is always index 0
        root_idx = 0
        
        # Level 1: children of root (indices 1, 2, 3 for branching=3)
        level1_start = 1
        level1_end = level1_start + len(self.root.child_nodes)
        level1_grads = grad_tensor[level1_start:level1_end]
        best_level1_local = level1_grads.argmax()
        best_level1_idx = level1_start + best_level1_local.item()
        
        # Level 2: children of best level1 node
        best_level1_node = self.all_nodes[best_level1_idx]
        if best_level1_node.child_nodes:
            # Find indices of children
            child_indices = [self.all_nodes.index(c) for c in best_level1_node.child_nodes]
            child_grads = grad_tensor[child_indices]
            best_child_local = child_grads.argmax()
            best_level2_idx = child_indices[best_child_local.item()]
        else:
            best_level2_idx = best_level1_idx
        
        # Path: [root, best_level1, best_level2]
        path_indices = [root_idx, best_level1_idx, best_level2_idx]
        
        # Create scale tensor (GPU) - all zeros except path nodes
        scales = torch.zeros(len(self.all_nodes), device=loss_tensor.device)
        scales[path_indices[-1]] = 1.0    # Culprit: 100%
        scales[path_indices[-2]] = 0.15   # Parent: 15%
        scales[path_indices[0]] = 0.05    # Root: 5%
        
        # Apply scales to gradients (GPU operation)
        for i, node in enumerate(self.all_nodes):
            scale = scales[i]
            for p in node.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        
        # Store for mark_nodes_updated (convert to dict only when needed)
        self._last_scales_indices = path_indices
    
    def _get_node_gradient_norm(self, node: TreeNode) -> float:
        """Get gradient norm for a specific node (uses cache if available)."""
        if node.node_id in self._grad_norm_cache:
            return self._grad_norm_cache[node.node_id]
        
        # Fallback: compute directly
        total = 0.0
        for p in node.parameters():
            if p.grad is not None:
                total += p.grad.norm().item()
        # NaN protection
        if math.isnan(total) or math.isinf(total):
            total = 0.0001
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
        """
        grad = self._get_node_gradient_norm(node)
        
        # NaN protection for loss
        if math.isnan(loss) or math.isinf(loss):
            loss = 1.0
        
        # Cooldown penalty: recently updated nodes get lower responsibility
        steps_since_update = current_step - node.last_updated_step
        if steps_since_update < cooldown_steps:
            cooldown_factor = penalty + (1 - penalty) * (steps_since_update / cooldown_steps)
        else:
            cooldown_factor = 1.0
        
        result = grad * loss * cooldown_factor
        
        # Final NaN protection
        if math.isnan(result) or math.isinf(result):
            result = 0.0001
        
        return result

    def find_responsible_path(
        self, 
        loss: float = 1.0, 
        current_step: int = 0
    ) -> List[Tuple[TreeNode, float]]:
        """
        Hierarchically drill down from Manager to find responsible agent path.
        
        Uses improved responsibility signal: gradient × loss × cooldown_penalty
        """
        path = []
        current = self.root

        while not current.is_leaf:
            resp = self.compute_responsibility(current, loss, current_step)
            path.append((current, resp))

            # Find child with highest responsibility (optimized: no list creation)
            if current.child_nodes:
                best_child = current.child_nodes[0]
                best_resp = self.compute_responsibility(best_child, loss, current_step)
                for child in current.child_nodes[1:]:
                    child_resp = self.compute_responsibility(child, loss, current_step)
                    if child_resp > best_resp:
                        best_child = child
                        best_resp = child_resp
                current = best_child

        # Add leaf (sub-agent culprit)
        path.append((current, self.compute_responsibility(current, loss, current_step)))
        return path

    def mark_nodes_updated(self, scales: dict, current_step: int):
        """Mark nodes that received updates (optimized: only iterate path)."""
        # Only update nodes that actually got scaled (usually 3)
        for node_id, scale in scales.items():
            if scale > 0:
                node = self.get_node_by_id(node_id)
                if node:
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
