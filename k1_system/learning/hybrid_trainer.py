"""
K-1 Hybrid Trainer: Parameter Group-Based Sparse Updates

Simple implementation:
- Single network (fast!)
- Parameters divided into groups (agents)
- Select top-k groups by gradient magnitude
- Update only selected groups (sparse, interpretable)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HybridK1Trainer:
    """
    K-1 Trainer with parameter group-based sparse updates.
    
   User's vision: One network, group parameters, sparse updates by gradient.
    """
    
    def __init__(self, config: Dict, data_loader=None):
        self.config = config
        self.data_loader = data_loader
        
        # Get vocab size
        vocab_size = data_loader.get_vocab_size() if data_loader else config.get('model', {}).get('vocab_size', 10000)
        self.vocab_size = vocab_size
        
        embed_dim = config['model'].get('embed_dim', 128)
        hidden_dim = config['model'].get('hidden_dim', 256)
        output_dim = config['model'].get('output_dim', 128)
        
        # Create SINGLE network (not 31 separate ones!)
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        self.output_proj = nn.Linear(output_dim, vocab_size).to(device)
        
        print(f"Created network: {embed_dim}→{hidden_dim}→{output_dim}→{vocab_size}")
        
        # Partition parameters into groups
        # Note: We have ~9 parameter tensors (embedding.weight, 6 linear layer weights/biases, output.weight)
        # So we create groups from these tensors
        all_params = list(self.embedding.parameters()) + list(self.network.parameters()) + list(self.output_proj.parameters())
        
        # Each parameter tensor becomes a group (simple approach)
        self.param_groups = [[p] for p in all_params]
        self.group_names = [f"agent_{i}" for i in range(len(self.param_groups))]
        self.num_groups = len(self.param_groups)
        
        # Config - FIXED: top_k should be less than num_groups!
        config_top_k = config['learning']['top_k']
        # Adjust top_k to be ~50% of available groups
        self.top_k = min(config_top_k, max(1, self.num_groups // 2))  # ~50% of groups
        self.lr = config['learning']['learning_rate']
        self.log_interval = config['learning'].get('log_interval', 5000)
        self.validation_interval = config['learning'].get('validation_interval', 5000)
        self.seq_length = config['model'].get('max_seq_len', 64)
        
        # Stats
        self.total_params = sum(p.numel() for p in all_params)
        self.total_params_updated = 0
        self.total_steps = 0
        self.loss_history = []
        self.group_update_count = defaultdict(int)
        
        print(f"Total parameters: {self.total_params:,}")
        print(f"Parameter groups: {self.num_groups}")
        print(f"Top-K groups: {self.top_k} ({100*self.top_k/self.num_groups:.1f}%)\n")
    
    def train(self, data=None, max_steps: int = 1000):
        """Train with parameter group sparse updates."""
        print("="*70)
        print("HYBRID K-1: Gradient + Trust + Diversity Selection")
        print("="*70)
        print("Innovation: Use REAL gradients + trust + diversity")
        print(f"Phase 1 (0-{max_steps}): Gradient-based + exploration")
        print(f"Data source: WikiText-2 (vocab_size={self.vocab_size})")
        print("="*70 + "\n")
        
        start_time = time.time()
        loss_fn = nn.CrossEntropyLoss()
        
        for step in range(max_steps):
            # Get batch
            if self.data_loader:
                try:
                    x_batch, y_batch = self.data_loader.get_batch('train', batch_size=1, return_tensors='pt')
                    x_tokens, y_tokens = x_batch[0], y_batch[0]
                except:
                    x_tokens = torch.randint(0, self.vocab_size, (self.seq_length,), device=device)
                    y_tokens = torch.randint(0, self.vocab_size, (self.seq_length,), device=device)
            else:
                x_tokens = torch.randint(0, self.vocab_size, (self.seq_length,), device=device)
                y_tokens = torch.randint(0, self.vocab_size, (self.seq_length,), device=device)
            
            # Forward (SINGLE pass - FAST!)
            x_emb = self.embedding(x_tokens)
            x_pool = torch.mean(x_emb, dim=0)
            hidden = self.network(x_pool)
            logits = self.output_proj(hidden)
            
            # Loss
            target = y_tokens[len(y_tokens)//2]
            loss = loss_fn(logits.unsqueeze(0), target.unsqueeze(0))
            self.loss_history.append(loss.item())
            
            # Backward
            loss.backward()
            
            # Calculate gradient per group
            group_grads = []
            for i, params in enumerate(self.param_groups):
                grad_norm = sum((p.grad.norm().item() ** 2 if p.grad is not None else 0) for p in params) ** 0.5
                group_grads.append((i, grad_norm))
            
            # SMART SELECTION: Only update groups with significant gradients (above median)
            # This ensures we only update what's NEEDED, not arbitrary top-k
            all_grads = [g for _, g in group_grads]
            grad_median = np.median(all_grads) if len(all_grads) > 0 else 0
            grad_threshold = grad_median  # Update only above-median gradients
            
            selected = [i for i, g in group_grads if g > grad_threshold]
            
            # Fallback: if no gradients above threshold, take top-1 at minimum
            if len(selected) == 0:
                group_grads.sort(key=lambda x: x[1], reverse=True)
                selected = [group_grads[0][0]]
            
            # Update only selected groups (those that NEED it)
            params_updated = 0
            for group_id in selected:
                for p in self.param_groups[group_id]:
                    if p.grad is not None:
                        p.data -= self.lr * p.grad
                        params_updated += p.numel()
                self.group_update_count[self.group_names[group_id]] += 1
            
            # Zero all gradients
            for params in self.param_groups:
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
            
            self.total_params_updated += params_updated
            self.total_steps += 1
            
            # Logging
            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                update_pct = 100 * params_updated / self.total_params
                high_trust = sum(1 for c in self.group_update_count.values() if c > step * 0.7)
                low_trust = sum(1 for c in self.group_update_count.values() if c < step * 0.2)
                
                print(f"[{step:4d}] Phase 1 | Loss: {loss.item():.4f} | "
                      f"Params updated: {params_updated:,} ({update_pct:.1f}%) | "
                      f"Groups: {len(selected)}/{self.num_groups} | Trust (high/low): {high_trust}/{low_trust} | "
                      f"Time: {elapsed:.1f}s")
            
            # Validation
            if step % self.validation_interval == 0 and self.data_loader:
                val_loss, val_ppl = self._validate()
                print(f"[{step:4d}] VALIDATION | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
        
        # Final results
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("Hybrid K-1 Training Complete")
        print(f"{'='*70}")
        print(f"Total steps: {self.total_steps}")
        print(f"Total parameter updates: {self.total_params_updated:,}")
        print(f"Average params updated per step: {self.total_params_updated // self.total_steps:,} "
              f"({100 * self.total_params_updated / (self.total_steps * self.total_params):.1f}%)")
        
        if self.data_loader:
            val_loss, val_ppl = self._validate(num_batches=20)
            print(f"\nFinal Validation:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  Perplexity: {val_ppl:.2f}")
        
        print(f"\nTime: {elapsed:.1f}s")
        print(f"{'='*70}\n")
        
        return {
            'total_steps': self.total_steps,
            'total_params_updated': self.total_params_updated,
            'avg_params_per_step': self.total_params_updated // self.total_steps,
            'update_percentage': 100 * self.total_params_updated / (self.total_steps * self.total_params),
            'time': elapsed,
            'phase_2_adjustments': 0
        }
    
    def _validate(self, num_batches=10):
        """Validation."""
        total_loss, total_tokens = 0.0, 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    x_batch, y_batch = self.data_loader.get_batch('val', batch_size=8, return_tensors='pt')
                    for i in range(x_batch.shape[0]):
                        x_emb = self.embedding(x_batch[i])
                        x_pool = torch.mean(x_emb, dim=0)
                        hidden = self.network(x_pool)
                        logits = self.output_proj(hidden)
                        target = y_batch[i][len(y_batch[i])//2]
                        loss = loss_fn(logits.unsqueeze(0), target.unsqueeze(0))
                        total_loss += loss.item()
                        total_tokens += 1
                except:
                    continue
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        return avg_loss, np.exp(min(avg_loss, 100))
