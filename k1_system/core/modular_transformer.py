"""
Modular Sparse Transformer for Sparse Parameter Updates

Architecture designed for sparse learning:
- Modular components (10 parameter groups)
- Skip connections (preserve gradient flow)
- Autoregressive language modeling loss
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class ModularSparseTransformer(nn.Module):
    """
    Transformer with modular design for sparse parameter updates.
    
    Architecture:
    - Embedding (Group 0)
    - 4 Transformer layers, each split into:
        - Multi-head attention (Groups 1, 3, 5, 7)
        - Feed-forward network (Groups 2, 4, 6, 8)
    - Output projection (Group 9)
    
    Total: 10 parameter groups
    
    Key features:
    - Skip connections for robustness
    - Modular components for interpretability
    - Works with sparse gradient-based updates
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 4, 
                 ff_dim: int = 256, max_seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Group 0: Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)
        
        # Groups 1-8: 4 transformer layers (attention + FFN per layer)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'attn': MultiHeadAttention(embed_dim, num_heads, dropout, max_seq_len),
                'attn_norm': nn.LayerNorm(embed_dim),
                'ffn': FeedForward(embed_dim, ff_dim, dropout),
                'ffn_norm': nn.LayerNorm(embed_dim)
            })
            self.layers.append(layer)
        
        # Group 9: Output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Initialize
        self._init_weights()
        
        print(f"Created ModularSparseTransformer:")
        print(f"  Vocab: {vocab_size}, Embed: {embed_dim}, Layers: {num_layers}")
        print(f"  Heads: {num_heads}, FF: {ff_dim}")
        print(f"  Parameter groups: {self.get_num_groups()}")
    
    def _init_weights(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass through modular transformer.
        
        Args:
            x: Input token indices [batch_size, seq_len] or [seq_len]
        
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size] or [seq_len, vocab_size]
        """
        # Handle single sequence or batch
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, seq_len]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = x.shape
        
        # Embedding + positional encoding
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)  # Add position
        
        # Pass through transformer layers with skip connections
        for layer in self.layers:
            # Multi-head attention with skip
            attn_out = layer['attn'](x, x, x)
            x = layer['attn_norm'](x + attn_out)  # Skip connection + norm
            
            # Feed-forward with skip
            ffn_out = layer['ffn'](x)
            x = layer['ffn_norm'](x + ffn_out)  # Skip connection + norm
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        
        if squeeze_output:
            logits = logits.squeeze(0)  # [seq_len, vocab_size]
        
        return logits
    
    def get_parameter_groups(self) -> List[List[nn.Parameter]]:
        """
        Get parameter groups for sparse updates.
        
        Returns:
            List of parameter groups (10 groups total)
        """
        groups = []
        
        # Group 0: Embedding
        groups.append(list(self.embedding.parameters()) + [self.pos_encoding])
        
        # Groups 1-8: Attention and FFN per layer
        for layer in self.layers:
            # Attention group
            attn_params = list(layer['attn'].parameters()) + list(layer['attn_norm'].parameters())
            groups.append(attn_params)
            
            # FFN group
            ffn_params = list(layer['ffn'].parameters()) + list(layer['ffn_norm'].parameters())
            groups.append(ffn_params)
        
        # Group 9: Output
        groups.append(list(self.output_norm.parameters()) + list(self.output_proj.parameters()))
        
        return groups
    
    def get_num_groups(self) -> int:
        """Get number of parameter groups"""
        return len(self.get_parameter_groups())


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 64):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Pre-compute scaling factor
        self.scale = self.head_dim ** -0.5

        # Pre-compute causal mask once (MAJOR SPEEDUP!)
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape for multi-head
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Use pre-computed causal mask (slice to current seq_len)
        if mask is None:
            mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Attention weights
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
