"""
TreeNode: Single node in the K-1 hierarchical tree.

Each node is a small transformer block that processes its input.
Nodes are organized in a tree structure: Root → Nodes → Agents → Sub-Agents.
"""

import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    pass  # Avoid circular imports


class TreeNode(nn.Module):
    """
    Single node in the hierarchical tree.
    Each node is a small transformer block that processes its input.
    
    Attributes:
        embed_dim: Embedding dimension
        node_id: Unique identifier for this node
        level: Depth level in tree (0=root, 1=nodes, 2=agents, 3=sub-agents)
        is_leaf: Whether this node has no children
        child_nodes: List of child nodes
        last_updated_step: Step when this node was last updated (for trust cooldown)
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        ff_dim: int, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        """
        Initialize a TreeNode.
        
        Args:
            embed_dim: Embedding dimension
            ff_dim: Feed-forward hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
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
        self.child_nodes = nn.ModuleList()  # CRITICAL: ModuleList for device transfer
        self.is_leaf = True
        
        # Tracking
        self.node_id: Optional[int] = None
        self.level: int = 0
        self.activation_count: int = 0
        self.gradient_norm: float = 0.0
        self.last_updated_step: int = -1000  # For trust cooldown
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through this node.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Processed tensor [batch, seq_len, embed_dim]
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.attn_norm(x + attn_out)
        
        # FFN with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)
        
        self.activation_count += 1
        return x
    
    def add_child(self, child: 'TreeNode'):
        """
        Add a child node.
        
        Args:
            child: Child TreeNode to add
        """
        self.child_nodes.append(child)
        self.is_leaf = False
        child.level = self.level + 1
    
    def get_gradient_norm(self) -> float:
        """
        Compute gradient norm for this node's parameters.
        
        Returns:
            L2 norm of all gradients in this node
        """
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.pow(2).sum().item()
        self.gradient_norm = math.sqrt(total_norm)
        return self.gradient_norm
    
    def __repr__(self) -> str:
        return (
            f"TreeNode(id={self.node_id}, level={self.level}, "
            f"is_leaf={self.is_leaf}, children={len(self.child_nodes)})"
        )
