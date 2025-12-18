"""
Baseline GPT Model (PyTorch) for fair comparison with K-1 Self-Learning System.

This implements a standard transformer-based language model using PyTorch
with GPU support, matching the K-1 model's parameter count (~135M).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with Flash Attention.
        Args:
            x: (batch, seq_len, embed_dim)
            mask: Optional causal mask
        Returns:
            (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use Flash Attention if available (PyTorch 2.0+)
        try:
            # scaled_dot_product_attention handles causal masking efficiently
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True  # Built-in causal masking
            )  # (B, heads, T, head_dim)
        except (AttributeError, RuntimeError):
            # Fallback to manual attention
            attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, T, T)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v  # (B, heads, T, head_dim)
        
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        # Pre-norm + feedforward + residual
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class BaselineGPTPyTorch(nn.Module):
    """
    Baseline GPT model for fair comparison with K-1 System.
    
    Scaled to ~135M parameters to match K-1 model.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model dimensions - scaled for ~135M params
        self.vocab_size = config.get('vocab_size', 1000)
        self.embed_dim = config.get('embed_dim', 512)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 8)
        self.ff_dim = config.get('ff_dim', 2048)
        self.max_seq_len = config.get('max_seq_len', 64)
        self.dropout = config.get('dropout', 0.1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)
        self.embed_dropout = nn.Dropout(self.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output head
        self.ln_final = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        
        # Weight tying (token embeddings = output projection)
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
        
        # Register causal mask buffer (after moving to device)
        self._register_causal_mask()
        
        # Compile model for faster execution (PyTorch 2.0+)
        try:
            # torch.compile can provide 2-3x speedup
            import torch._dynamo
            self._compiled = True
        except ImportError:
            self._compiled = False
        
        # Optimizer (Adam with weight decay)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.step_count = 0
        
        # Training state
        self.total_loss = 0.0
        self.loss_history = []
        
    def _init_weights(self, module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def _register_causal_mask(self):
        """Register causal attention mask as buffer."""
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, device=self.device))
        self.register_buffer('causal_mask', mask.view(1, 1, self.max_seq_len, self.max_seq_len))
        
    def _get_lr(self) -> float:
        """Get learning rate with linear warmup."""
        if self.step_count < self.warmup_steps:
            return self.optimizer.defaults['lr'] * (self.step_count + 1) / self.warmup_steps
        return self.optimizer.defaults['lr']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Token indices (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        
        # Get embeddings
        tok_emb = self.token_embedding(x)  # (B, T, C)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embedding(pos)  # (T, C)
        
        x = self.embed_dropout(tok_emb + pos_emb)
        
        # Get causal mask for this sequence length
        mask = self.causal_mask[:, :, :T, :T]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_final(x)
        logits = self.output_proj(x)
        
        return logits
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Single training step with gradient clipping.
        Args:
            x: Input tokens (batch, seq_len)
            y: Target tokens (batch, seq_len)
        Returns:
            loss value
        """
        self.train()
        self.step_count += 1
        
        # Move to device if needed
        if not x.is_cuda and self.device.type == 'cuda':
            x = x.to(self.device)
            y = y.to(self.device)
        
        # Update learning rate for warmup
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr()
        
        # Forward pass
        self.optimizer.zero_grad()
        logits = self.forward(x)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            y.view(-1)
        )
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        loss_val = loss.item()
        self.total_loss += loss_val
        self.loss_history.append(loss_val)
        
        return loss_val
    
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50, 
                 temperature: float = 1.0, top_k: int = 50) -> List[int]:
        """
        Generate tokens autoregressively.
        Args:
            prompt: Starting token indices (can be numpy array or tensor)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
        Returns:
            List of generated token indices
        """
        self.eval()
        
        # Handle numpy input
        if hasattr(prompt, 'numpy'):
            prompt = prompt.cpu().numpy() if hasattr(prompt, 'cpu') else prompt
        if isinstance(prompt, (list, tuple)) or (hasattr(prompt, '__len__') and not isinstance(prompt, torch.Tensor)):
            import numpy as np
            prompt = np.array(prompt)
            
        if isinstance(prompt, torch.Tensor):
            tokens = prompt.tolist()
        else:
            tokens = list(prompt)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get context (last max_seq_len tokens)
                context = tokens[-self.max_seq_len:]
                x = torch.tensor([context], dtype=torch.long, device=self.device)
                
                # Forward pass
                logits = self.forward(x)
                next_logits = logits[0, -1] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < values[-1]] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token)
        
        return tokens
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'total_parameters': total_params,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'step_count': self.step_count,
            'avg_loss': self.total_loss / max(1, self.step_count),
            'recent_loss': sum(self.loss_history[-100:]) / max(1, len(self.loss_history[-100:])) if self.loss_history else 0.0,
            'device': str(self.device)
        }
    
    def get_current_phase(self) -> str:
        """Get training phase (for compatibility with K-1 model)."""
        return "Transformer"


def create_baseline_model(config: Dict) -> BaselineGPTPyTorch:
    """Factory function to create baseline model."""
    return BaselineGPTPyTorch(config)
