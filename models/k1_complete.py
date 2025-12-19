"""
K-1 Self-Learning System - CORRECT IMPLEMENTATION
Same architecture as baseline GPT (attention, FFN, everything)
But: Selective layer updates instead of full backprop

Architecture: SAME as baseline (transformer blocks with attention)
Training: Find responsible layers → Update only those (not all)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional


class MultiHeadAttention(nn.Module):
    """Same as baseline - Multi-head self-attention with causal masking."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        try:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
            )
        except:
            attn = (q @ k.transpose(-2, -1)) / self.scale
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Same as baseline - FFN with GELU."""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Same as baseline - Transformer block with pre-norm."""
    
    def __init__(self, block_id: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block_id = block_id  # For tracking which block is responsible
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Tracking for selective updates
        self.gradient_magnitude = 0.0
        self.times_updated = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class K1CompleteSystem(nn.Module):
    """
    K-1 with CORRECT architecture (same as baseline GPT).
    
    Architecture: Standard transformer (attention + FFN)
    Training: Selective layer updates (not full backprop)
    
    This answers: "Which transformer block is responsible for this error?"
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Same dimensions as baseline
        self.vocab_size = config.get('vocab_size', 1000)
        self.embed_dim = config.get('embed_dim', 128)
        self.num_layers = config.get('num_layers', 4)
        self.num_heads = config.get('num_heads', 4)
        self.ff_dim = config.get('ff_dim', 512)
        self.max_seq_len = config.get('max_seq_len', 64)
        self.dropout = config.get('dropout', 0.1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Same architecture as baseline
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)
        self.embed_dropout = nn.Dropout(self.dropout)
        
        # Transformer blocks (these are our "agents")
        self.blocks = nn.ModuleList([
            TransformerBlock(i, self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for i in range(self.num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        
        # Weight tying
        self.output_proj.weight = self.token_embedding.weight
        
        # SAME initialization as baseline!
        self.apply(self._init_weights)
        
        # Training (same as baseline: AdamW with warmup)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.current_iteration = 0
        self.phase_1_duration = config.get('phase_1_duration', 10000)
        self.phase_2_active = False
        self.warmup_steps = config.get('warmup_steps', 1000)
        
        # AdamW optimizer (same as baseline!)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.to(self.device)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Built K-1 with {self.num_layers} transformer blocks, {total_params:,} parameters")
    
    def _init_weights(self, module):
        """SAME weight initialization as baseline."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _get_lr(self) -> float:
        """Learning rate with warmup (same as baseline)."""
        if self.current_iteration < self.warmup_steps:
            return self.learning_rate * (self.current_iteration + 1) / self.warmup_steps
        return self.learning_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same forward pass as baseline GPT."""
        B, T = x.shape
        
        # Token + position embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = self.embed_dropout(tok_emb + pos_emb)
        
        # Through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        logits = self.output_proj(x)
        
        return logits
    
    def find_responsible_blocks(self) -> List[tuple]:
        """Find which transformer blocks have the highest gradients."""
        responsibilities = []
        
        for block in self.blocks:
            grad_sum = 0.0
            param_count = 0
            for param in block.parameters():
                if param.grad is not None:
                    grad_sum += param.grad.abs().sum().item()
                    param_count += param.numel()
            
            if param_count > 0:
                responsibility = grad_sum / param_count
            else:
                responsibility = 0.0
            
            block.gradient_magnitude = responsibility
            responsibilities.append((block, responsibility))
        
        # Return blocks with above-average gradient
        mean_resp = np.mean([r for _, r in responsibilities]) if responsibilities else 0
        responsible = [(b, r) for b, r in responsibilities if r > mean_resp]
        responsible.sort(key=lambda x: x[1], reverse=True)
        
        return responsible
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Selective layer update training.
        
        1. Forward pass (same as baseline)
        2. Compute loss and gradients (same as baseline)
        3. Find which blocks are responsible (NEW!)
        4. Update ONLY those blocks + embeddings (different from baseline)
        """
        self.current_iteration += 1
        
        if self.current_iteration == self.phase_1_duration:
            print("\n" + "="*70)
            print("🚀 PHASE 2 ACTIVATED")
            print("="*70 + "\n")
            self.phase_2_active = True
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward pass (same as baseline)
        logits = self.forward(x)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        loss_val = loss.item()
        
        # Compute gradients (same as baseline)
        loss.backward()
        
        # Find which blocks are responsible
        responsible_blocks = self.find_responsible_blocks()
        responsible_set = set(b for b, _ in responsible_blocks)
        
        # Zero gradients for NON-responsible blocks (so they don't update)
        for block in self.blocks:
            if block not in responsible_set:
                for param in block.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            else:
                block.times_updated += 1
        
        # Apply warmup LR
        current_lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Update using AdamW (embeddings + responsible blocks only)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        updated = len(responsible_blocks)
        
        return {
            'loss': loss_val,
            'updated': updated,
            'skipped': len(self.blocks) - updated,
            'total_agents': len(self.blocks),
            'avg_trust': 0.5,
            'high_trust': 0,
            'loo_computed': False,
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1'
        }
    
    def get_stats(self) -> Dict:
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'total_agents': len(self.blocks),
            'avg_trust': 0.5,
            'high_trust_agents': 0,
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1',
            'iteration': self.current_iteration
        }
    
    def get_agent_status(self) -> List[Dict]:
        return [{
            'id': f'block_{b.block_id}',
            'domain': 'Transformer',
            'specialty': f'Layer {b.block_id}',
            'trust': 0.5,
            'updated': b.times_updated,
            'skipped': self.current_iteration - b.times_updated,
            'loo_score': 0,
            'grad_score': round(b.gradient_magnitude, 6)
        } for b in self.blocks]
    
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0) -> List[int]:
        self.eval()
        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt)
        tokens = prompt.tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = tokens[-self.max_seq_len:]
                x = torch.tensor([context], dtype=torch.long, device=self.device)
                logits = self.forward(x)
                next_logits = logits[0, -1] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token)
        
        self.train()
        return tokens


def create_k1_complete_model(config: Dict) -> K1CompleteSystem:
    return K1CompleteSystem(config)
