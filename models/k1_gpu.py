import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
from core.gpu_hierarchy import GPUHierarchy

class K1GPUModel(nn.Module):
    """
    K-1 Self-Learning Model (GPU Optimized).
    
    Implements the "Vectorized Forward" strategy:
    - Parallel processing of batches
    - Index-based routing
    - Autograd for backpropagation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dimensions
        self.vocab_size = config.get('vocab_size', 1000)
        self.embed_dim = config.get('embed_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Hierarchy Config
        depth = config.get('hierarchy_depth', 3)
        children = config.get('branching_factor', 4)
        
        # Components
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim).to(self.device)
        self.hierarchy = GPUHierarchy(
            max_agents=2000,
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embed_dim,
            max_children=children,
            device=self.device
        )
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size).to(self.device)
        
        # Training State
        self.max_depth = depth
        self.optimizer = optim.Adam(self.parameters(), lr=config.get('learning_rate', 1e-4))
        
        # Initial Structure Build (Simple binary tree for test)
        self._build_initial_tree_gpu(depth, children)
        
    def _build_initial_tree_gpu(self, depth, children):
        """Build a basic tree structure in the tensor hierarchy."""
        # Simple BFS expansion
        queue = [(0, 0)] # (idx, current_depth)
        
        while queue:
            idx, curr_d = queue.pop(0)
            if curr_d < depth:
                for _ in range(children):
                    child_idx = self.hierarchy.add_child(idx)
                    if child_idx != -1:
                        queue.append((child_idx, curr_d + 1))
                        
    def forward(self, x_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x_indices: (Batch_Size, Seq_Len) or (Batch_Size,)
        """

        if isinstance(x_indices, np.ndarray):
             x_indices = torch.from_numpy(x_indices).long().to(self.device)

        if x_indices.dim() == 2:
            # Flatten sequence: treat (Batch, Seq) as (Batch*Seq)
            B, T = x_indices.shape
            x_flat = x_indices.view(-1) # (B*T)
        else:
            x_flat = x_indices
            B = x_indices.shape[0]
            T = 1
            
        # 1. Embeddings
        x_emb = self.embedding(x_flat) # (Batch, Embed)
        
        # 2. Hierarchy Traversal
        # Start at Root (Index 0) for everyone
        curr_agents = torch.zeros(x_flat.shape[0], dtype=torch.long, device=self.device)
        
        final_hidden = torch.zeros_like(x_emb)
        
        # We loop through depths defined by max_depth
        # In a real dynamic routing, we stops when leaf is reached
        for d in range(self.max_depth + 1):
            
            # Run Agent
            h, out, r_logits = self.hierarchy.forward_batch_agents(curr_agents, x_emb)
            
            # Accumulate output (Residual connection style or overwrite)
            # For simplicity, let's use the output of the final agent
            final_hidden = out
            
            # Decide Next Child (Hard Routing)
            # greedy: argmax of router logits
            # r_logits: (Batch, Max_Children)
            best_child_local_idx = torch.argmax(r_logits, dim=1) # (Batch,)
            
            # Map local child index (0..3) to global agent index
            # children_table: (Batch, Max_Children)
            
            # Safety Check: Ensure curr_agents are within bounds
            curr_agents = torch.clamp(curr_agents, 0, self.hierarchy.max_agents - 1)
            
            children_table = self.hierarchy.get_children(curr_agents) 
            
            # Gather the global index
            # new_agents = children_table[batch_idx, best_child_local_idx]
            new_agents = torch.gather(children_table, 1, best_child_local_idx.unsqueeze(1)).squeeze(1)
            
            # Mask out invalid children (-1)
            # If invalid, stay at current agent (or stop)
            mask_valid = (new_agents != -1)
            curr_agents = torch.where(mask_valid, new_agents, curr_agents)
            
            # If all invalid, we could break early, but batching makes that hard
            
        # 3. Output Projection
        logits = self.output_proj(final_hidden)
        
        return logits.view(B, T, -1) if T > 1 else logits

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compatibility wrapper for training."""
        self.train()
        
        x_t = torch.tensor(x, dtype=torch.long, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # Unsqueeze for batch dim if needed
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            y_t = y_t.unsqueeze(0)
            
        self.optimizer.zero_grad()
        
        logits = self.forward(x_t)
        
        # Reshape for loss (Batch*Seq, Vocab)
        loss = nn.functional.cross_entropy(logits.reshape(-1, self.vocab_size), y_t.reshape(-1))
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def get_current_phase(self) -> str:
        """Get current training phase."""
        return "GPU Optimized"
        
    def get_stats(self):
        """Mock stats for compatibility."""
        return {
            'num_agents': self.hierarchy.agent_count,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'phase': 'GPU Optimized'
        }
        
    def generate(self, prompt, max_new_tokens=50):
        """Generation (Basic greedy)."""
        self.eval()
        curr = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        
        res = list(prompt)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(curr)
                next_token = torch.argmax(logits[0, -1]).item()
                res.append(next_token)
                curr = torch.tensor([res], dtype=torch.long, device=self.device)
        return res
