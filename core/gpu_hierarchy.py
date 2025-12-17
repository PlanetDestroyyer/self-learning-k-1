"""
GPU-Optimized Hierarchy for K-1 System.

Replaces object-based hierarchy with monolithic PyTorch tensors for massive parallelism.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class GPUHierarchy(nn.Module):
    """
    Vectorized Hierarchy Manager.
    
    Instead of Agent objects, we store huge tensors:
    - W1: (Max_Agents, In_Dim, Hidden_Dim)
    - b1: (Max_Agents, Hidden_Dim)
    - W2: (Max_Agents, Hidden_Dim, Out_Dim)
    - b2: (Max_Agents, Out_Dim)
    - Router: (Max_Agents, Hidden_Dim, Max_Children)
    
    Tree Structure is stored as indices:
    - Child_Indices: (Max_Agents, Max_Children) -> Index of child in the big tensor
    """
    
    def __init__(
        self,
        max_agents: int = 1000,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        max_children: int = 8,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.max_agents = max_agents
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_children = max_children
        self.device = device
        
        # === 1. Neural Network Weights (Monolithic Tensors) ===
        # We wrap them in nn.Parameter logic later or manual management
        # For now, let's use nn.ParameterList or just big Parameters
        
        # Layer 1
        self.W1 = nn.Parameter(torch.randn(max_agents, input_dim, hidden_dim, device=device) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(max_agents, hidden_dim, device=device))
        
        # Layer 2
        self.W2 = nn.Parameter(torch.randn(max_agents, hidden_dim, output_dim, device=device) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(max_agents, output_dim, device=device))
        
        # Router (maps hidden state to child logits)
        self.router_weights = nn.Parameter(
            torch.randn(max_agents, hidden_dim, max_children, device=device) * 0.01
        )
        
        # === 2. Structural Indices (Int Tensors) ===
        # -1 indicates "No Child"
        self.child_indices = torch.full((max_agents, max_children), -1, dtype=torch.long, device=device)
        self.active_mask = torch.zeros(max_agents, dtype=torch.bool, device=device)
        self.depths = torch.zeros(max_agents, dtype=torch.long, device=device)
        
        # === 3. State Management ===
        self.agent_count = 0
        self.root_idx = 0
        
        # Initialize Root
        self._init_agent(0, depth=0)
        self.active_mask[0] = True
        self.agent_count = 1
        
    def _init_agent(self, idx: int, depth: int):
        """Initialize parameters for a new agent slot."""
        # Reset weights to nice initialization
        nn.init.xavier_uniform_(self.W1[idx])
        nn.init.zeros_(self.b1[idx])
        nn.init.xavier_uniform_(self.W2[idx])
        nn.init.zeros_(self.b2[idx])
        nn.init.normal_(self.router_weights[idx], std=0.01)
        
        self.child_indices[idx] = -1
        self.depths[idx] = depth
        
    def add_child(self, parent_idx: int) -> int:
        """
        Create a new child for parent_idx.
        Returns: New child index
        """
        if self.agent_count >= self.max_agents:
            raise RuntimeError("Hierarchy Full! Increase max_agents")
            
        # Find next empty slot
        child_idx = self.agent_count
        self.agent_count += 1
        
        # Initialize
        parent_depth = self.depths[parent_idx].item()
        self._init_agent(child_idx, depth=parent_depth + 1)
        self.active_mask[child_idx] = True
        
        # Link to parent
        # Find empty slot in parent's child list
        slots = self.child_indices[parent_idx]
        empty_slot = (slots == -1).nonzero(as_tuple=True)[0]
        
        if len(empty_slot) == 0:
            # Parent is full, maybe expand max_children? 
            # For GPU simplicity, we just fail or ignore for now
            return -1
            
        slot = empty_slot[0].item()
        self.child_indices[parent_idx, slot] = child_idx
        
        return child_idx

    def forward_batch_agents(self, 
                             agent_indices: torch.Tensor, 
                             x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass for a BATCH of agents.
        
        Args:
            agent_indices: (Batch_Size,) - Which agent to run for each sample
            x: (Batch_Size, Input_Dim) - Input for each sample
            
        Returns:
            hidden: (Batch_Size, Hidden_Dim)
            output: (Batch_Size, Output_Dim)
            router_logits: (Batch_Size, Max_Children)
        """
        B = x.shape[0]
        
        # 1. Gather Wrappers (The Magic Step)
        # We pick the specific W1/b1 tensors for the agents requested
        # W1[indices] shape: (Batch, In, Hidden)
        w1_curr = self.W1[agent_indices] 
        b1_curr = self.b1[agent_indices]
        w2_curr = self.W2[agent_indices]
        b2_curr = self.b2[agent_indices]
        r_curr  = self.router_weights[agent_indices]
        
        # 2. Batch Matrix Multiplication (Parallel Compute)
        # x is (B, In), needs to be (B, 1, In)
        x_uns = x.unsqueeze(1)
        
        # h = x @ W1 + b1
        # (B,1,In) @ (B,In,Hidden) -> (B,1,Hidden)
        h_pre = torch.bmm(x_uns, w1_curr).squeeze(1) + b1_curr
        h = torch.relu(h_pre)
        
        # out = h @ W2 + b2
        out = torch.bmm(h.unsqueeze(1), w2_curr).squeeze(1) + b2_curr
        
        # router = h @ R
        r_logits = torch.bmm(h.unsqueeze(1), r_curr).squeeze(1)
        
        return h, out, r_logits

    def get_children(self, agent_indices: torch.Tensor) -> torch.Tensor:
        """
        Get child indices for a batch of agents.
        Returns: (Batch, Max_Children)
        """
        return self.child_indices[agent_indices]
