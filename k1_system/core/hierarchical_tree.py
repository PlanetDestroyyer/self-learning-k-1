"""
K-1 Hierarchical Tree for Language Modeling

Architecture:
                    ROOT (Manager)
                         |
         ┌───────────────┼───────────────┐
         |               |               |
      Node 1          Node 2          Node 3
         |               |               |
    ┌────┼────┐     ┌────┼────┐    ┌────┼────┐
   L1   L2   L3    L4   L5   L6   L7   L8   L9

Data flows DOWN, gradients flow UP.
Update only the PATH responsible for errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TreeNode(nn.Module):
    """
    Single node in the hierarchical tree.
    Each node is a small transformer block that processes its input.
    """
    
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Self-attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
        
        # Router to children (if not leaf)
        self.router = None  # Set by parent
        self.child_nodes: List['TreeNode'] = []  # Renamed to avoid PyTorch conflict
        self.is_leaf = True
        
        # For tracking
        self.node_id = None
        self.level = 0
        self.activation_count = 0
        self.gradient_norm = 0.0
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process input through this node."""
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.attn_norm(x + attn_out)
        
        # FFN with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)
        
        self.activation_count += 1
        return x
    
    def add_child(self, child: 'TreeNode'):
        """Add a child node."""
        self.child_nodes.append(child)
        self.is_leaf = False
        child.level = self.level + 1
    
    def get_gradient_norm(self) -> float:
        """Compute gradient norm for this node's parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.pow(2).sum().item()
        self.gradient_norm = math.sqrt(total_norm)
        return self.gradient_norm


class HierarchicalTree(nn.Module):
    """
    Hierarchical tree of nodes for language modeling.
    
    Structure:
    - Shared embedding layer
    - Tree of nodes (depth configurable)
    - Shared output projection
    
    Key innovation: Path-based gradient updates
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        ff_dim: int = 256,
        num_heads: int = 4,
        tree_depth: int = 4,          # Root + Nodes + Agents + Sub-Agents
        branching_factor = None,      # [4, 3, 2] or single int
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tree_depth = tree_depth

        # Support variable branching: [4 nodes, 3 agents, 2 sub-agents]
        if branching_factor is None:
            branching_factor = [4, 3, 2]  # Default: 4 Nodes, 3 Agents, 2 Sub-Agents
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
        node = TreeNode(self.embed_dim, self.embed_dim * 2)
        node.node_id = node_id
        node.level = depth

        if depth < max_depth - 1:
            # Get branching factor for this level
            if isinstance(self.branching_factor, list):
                num_children = self.branching_factor[depth] if depth < len(self.branching_factor) else self.branching_factor[-1]
            else:
                num_children = self.branching_factor

            # Create children
            for i in range(num_children):
                # Generate unique child ID
                child_id = len(self.all_nodes) if hasattr(self, 'all_nodes') else node_id * 10 + i + 1
                child = self._build_tree(depth + 1, max_depth, child_id)
                node.add_child(child)

        return node
    
    def _collect_nodes(self, node: TreeNode):
        """Collect all nodes into a list."""
        self.all_nodes.append(node)
        for child in node.child_nodes:
            self._collect_nodes(child)
    
    def _init_weights(self):
        """Initialize weights."""
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
        
        # Process through tree (all nodes for now, routing later)
        path = []
        h = self._forward_node(h, self.root, mask, path)
        
        # Output projection
        h = self.output_norm(h)
        logits = self.output_proj(h)
        
        return logits, path
    
    def _forward_node(self, x: torch.Tensor, node: TreeNode, mask: torch.Tensor, path: List[TreeNode]) -> torch.Tensor:
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
    
    def get_path_gradient_norms(self) -> dict:
        """Get gradient norms for all nodes."""
        norms = {}
        for node in self.all_nodes:
            norms[node.node_id] = node.get_gradient_norm()
        return norms
    
    def get_node_by_id(self, node_id: int) -> Optional[TreeNode]:
        """Get node by ID."""
        for node in self.all_nodes:
            if node.node_id == node_id:
                return node
        return None

    def _get_node_gradient_norm(self, node: TreeNode) -> float:
        """Get gradient norm for a specific node."""
        total = 0.0
        for p in node.parameters():
            if p.grad is not None:
                total += p.grad.norm().item()
        return total

    def find_responsible_path(self) -> List[Tuple[TreeNode, float]]:
        """
        Hierarchically drill down from Manager to find responsible agent path.

        Returns:
            List of (node, gradient_norm) from Manager → Agent → Sub-Agent
        """
        path = []
        current = self.root

        while not current.is_leaf:
            grad = self._get_node_gradient_norm(current)
            path.append((current, grad))

            # Find child with highest gradient (responsible child)
            if current.child_nodes:
                child_grads = [
                    (child, self._get_node_gradient_norm(child))
                    for child in current.child_nodes
                ]
                culprit_child, _ = max(child_grads, key=lambda x: x[1])
                current = culprit_child

        # Add leaf (sub-agent culprit)
        path.append((current, self._get_node_gradient_norm(current)))
        return path

    def get_proportional_scales(self, responsible_path: List[Tuple[TreeNode, float]]) -> dict:
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
            if node == path_nodes[-1]:  # Culprit (deepest in path - sub-agent)
                scales[node.node_id] = 1.0  # 100% update
            elif len(path_nodes) > 1 and node == path_nodes[-2]:  # Parent (agent)
                scales[node.node_id] = 0.15  # 15% update
            elif node == path_nodes[0]:  # Manager (root)
                scales[node.node_id] = 0.05  # 5% update
            else:
                scales[node.node_id] = 0.0  # Not in responsible path, skip

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

    def print_responsibility_tree(self, grad_norms: dict, scales: dict, node: TreeNode = None, level: int = 0):
        """
        Print hierarchical responsibility visualization.

        Args:
            grad_norms: Dict of node_id -> gradient_norm
            scales: Dict of node_id -> update_scale
            node: Current node (defaults to root)
            level: Current tree level
        """
        if node is None:
            node = self.root

        indent = "  " * level
        node_id = node.node_id
        grad = grad_norms.get(node_id, 0.0)
        scale = scales.get(node_id, 0.0)

        # Status icon based on update scale
        if scale >= 0.8:
            icon = "🚨"  # Culprit (Sub-Agent)
        elif scale > 0:
            icon = "⚠️"   # Parent/Manager
        else:
            icon = "✓"    # OK, not responsible

        # Role name based on level
        if level == 0:
            role = "Root   "  # Hidden root
        elif level == 1:
            role = "Node   "  # Top-level Nodes
        elif level == 2:
            role = "Agent  "  # Agents within Nodes
        else:
            role = "SubAgent"  # Sub-Agents within Agents

        print(f"{indent}{icon} {role} {node_id}: grad={grad:.3f}, update={scale*100:3.0f}%")

        # Recursively print children
        for child in node.child_nodes:
            self.print_responsibility_tree(grad_norms, scales, child, level + 1)


class HierarchicalK1Trainer:
    """
    Trainer for hierarchical K-1 system.
    
    Key innovation: Path-based gradient updates.
    - Compute gradients for entire tree
    - Identify high-gradient paths (responsible for errors)
    - Update those paths more, others less
    """
    
    def __init__(self, config: dict, data_loader=None):
        self.config = config
        self.data_loader = data_loader
        
        vocab_size = data_loader.get_vocab_size() if data_loader else 10000
        embed_dim = config['model'].get('embed_dim', 128)
        ff_dim = config['model'].get('hidden_dim', 256)
        num_heads = config['model'].get('num_heads', 4)
        tree_depth = config['model'].get('tree_depth', 3)
        branching_factor = config['model'].get('branching_factor', 3)
        max_seq_len = config['model'].get('max_seq_len', 64)
        
        # Create hierarchical model
        self.model = HierarchicalTree(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            tree_depth=tree_depth,
            branching_factor=branching_factor,
            max_seq_len=max_seq_len
        ).to(device)
        
        # Training settings
        self.lr = config['learning'].get('learning_rate', 0.001)
        self.top_k_nodes = config['learning'].get('top_k', 5)  # Update top-k nodes
        self.log_interval = config['learning'].get('log_interval', 100)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        # AMP
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        
        # Stats
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.vocab_size = vocab_size
        self.seq_length = max_seq_len
        
        print(f"Total parameters: {self.total_params:,}")
        print(f"Nodes in tree: {len(self.model.all_nodes)}")
        print(f"Top-K nodes to update: {self.top_k_nodes}")
    
    def train(self, max_steps: int = 1000):
        """Train with path-based gradient updates."""
        import time
        
        print("=" * 70)
        print("HIERARCHICAL K-1: Path-Based Gradient Updates")
        print("=" * 70)
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Tree structure: depth={self.model.tree_depth}, branching={self.model.branching_factor}")
        print("=" * 70)
        
        loss_fn = nn.CrossEntropyLoss()
        batch_size = self.config['learning'].get('batch_size', 32)
        start_time = time.time()
        total_loss = 0.0
        
        for step in range(max_steps):
            # Get batch
            if self.data_loader:
                try:
                    x, y = self.data_loader.get_batch('train', batch_size=batch_size, return_tensors='pt')
                except:
                    x = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
                    y = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
            else:
                x = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
                y = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
            
            # Forward pass
            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, path = self.model(x)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    y[:, 1:].reshape(-1)
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # ============================================
            # HIERARCHICAL ERROR ATTRIBUTION
            # ============================================
            with torch.no_grad():
                # Step 1: Hierarchically drill down to find responsible path
                #         Manager → Agent X → Sub-Agent Y
                responsible_path = self.model.find_responsible_path()

                # Step 2: Compute proportional update scales
                #         Sub-Agent: 100%, Agent: 15%, Manager: 5%, Others: 0%
                scales = self.model.get_proportional_scales(responsible_path)

                # Step 3: Apply scaled gradients (proportional updates!)
                self.model.apply_proportional_updates(scales)

                # Track which nodes are updated
                nodes_to_update = [nid for nid, scale in scales.items() if scale > 0]

                # Compute gradient norms for logging
                grad_norms = {
                    node.node_id: self.model._get_node_gradient_norm(node)
                    for node in self.model.all_nodes
                }

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate loss WITHOUT sync (CRITICAL!)
            total_loss += loss.detach()
            
            # Logging
            if step % self.log_interval == 0:
                # Single GPU-CPU sync per log interval
                avg_loss = (total_loss / (step + 1)).item()
                elapsed = time.time() - start_time
                speed = (step + 1) / elapsed if elapsed > 0 else 0

                # Print hierarchical error attribution
                print(f"\n[{step:6d}] Loss: {avg_loss:.4f} | Speed: {speed:.1f} step/s")
                print("─" * 60)
                print("Hierarchical Error Attribution:")
                self.model.print_responsibility_tree(grad_norms, scales)

                # Show responsible path
                path_str = " → ".join(
                    f"Node{node.node_id}(g={grad:.2f})"
                    for node, grad in responsible_path
                )
                print(f"\nError Path: {path_str}")

                # Summary
                num_updated = len(nodes_to_update)
                num_total = len(self.model.all_nodes)
                pct_updated = (num_updated / num_total * 100) if num_total > 0 else 0
                print(f"Updated: {num_updated}/{num_total} nodes ({pct_updated:.0f}%) | "
                      f"Preserved: {num_total - num_updated} nodes ({100-pct_updated:.0f}%)")
                print("─" * 60)
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete: {max_steps} steps in {elapsed:.1f}s")
        
        return {'loss': total_loss / max_steps, 'time': elapsed}
