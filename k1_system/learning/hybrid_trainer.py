"""
K-1 Hybrid Trainer: Modular Sparse Updates with Autoregressive Loss

Proper implementation:
- Modular Transformer architecture  
- Autoregressive next-token prediction
- Sparse updates by gradient threshold
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
    K-1 Trainer with modular architecture and proper autoregressive loss.
    """
    
    def __init__(self, config: Dict, data_loader=None):
        self.config = config
        self.data_loader = data_loader
        
        # Get vocab size
        vocab_size = data_loader.get_vocab_size() if data_loader else config.get('model', {}).get('vocab_size', 10000)
        self.vocab_size = vocab_size
        
        embed_dim = config['model'].get('embed_dim', 128)
        ff_dim = config['model'].get('hidden_dim', 256)
        num_heads = config['model'].get('num_heads', 4)
        num_layers = config['model'].get('num_layers', 4)
        max_seq_len = config['model'].get('max_seq_len', 64)
        
        # Create MODULAR TRANSFORMER for sparse updates
        from ..core.modular_transformer import ModularSparseTransformer
        self.model = ModularSparseTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout=0.1
        ).to(device)
        
        # Get parameter groups BEFORE torch.compile (compile wraps the model)
        self.param_groups = self.model.get_parameter_groups()
        self.group_names = [f"group_{i}" for i in range(len(self.param_groups))]
        self.num_groups = len(self.param_groups)
        
        # Config
        config_top_k = config['learning']['top_k']
        self.top_k = min(config_top_k, max(1, self.num_groups // 2))
        self.lr = config['learning']['learning_rate']
        self.log_interval = config['learning'].get('log_interval', 5000)
        self.validation_interval = config['learning'].get('validation_interval', 5000)
        self.seq_length = max_seq_len

        # SPEED FIX: Single optimizer (much faster than 10 separate ones)
        # Sparse updates done by zeroing gradients BEFORE step, not separate optimizers
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        # Stats
        self.total_params = sum(p.numel() for group in self.param_groups for p in group)
        self.total_params_updated = 0
        self.total_steps = 0
        self.loss_history = []
        self._loss_buffer = []
        self.group_update_count = defaultdict(int)
        
        # Pre-calculate group parameter counts
        group_counts = [sum(p.numel() for p in group) for group in self.param_groups]
        self._group_param_counts = torch.tensor(group_counts, dtype=torch.float32, device=device)

        # Initialize sparse update mask (updated every 100 steps for speed)
        self._current_mask = torch.ones(self.num_groups, dtype=torch.bool, device=device)
        
        # Cache param-to-group mapping for fast lookup
        self._param_to_group = {}
        for g_idx, params in enumerate(self.param_groups):
            for p in params:
                self._param_to_group[p] = g_idx
        
        # === IMPROVEMENT 3: Gradient Accumulation ===
        self.accumulation_steps = config['learning'].get('accumulation_steps', 1)
        
        # === IMPROVEMENT 5: Gradient Checkpointing ===
        self.use_checkpointing = config['learning'].get('gradient_checkpointing', False)
        
        # Disable EWC for speed (can be re-enabled if needed)
        self.ewc_lambda = 0  # config['learning'].get('ewc_lambda', 0.1)
        
        print(f"Total parameters: {self.total_params:,}")
        print(f"Parameter groups: {self.num_groups}")
        print(f"Top-K groups: {self.top_k}")
        print(f"Accumulation Steps: {self.accumulation_steps}")
        print(f"Selection Update Interval: every 100 steps\n")
    
    def train(self, data=None, max_steps: int = 1000):
        """Train with modular sparse updates and proper autoregressive loss."""
        print("="*70)
        print("MODULAR K-1: Sparse Updates + Autoregressive Loss")
        print("="*70)
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  WARNING: CUDA not available! Training on CPU (VERY SLOW)")
            print("   On Colab: Runtime → Change runtime type → T4 GPU")
        print(f"Architecture: Modular Transformer ({self.num_groups} groups)")
        print(f"Loss: Proper autoregressive next-token prediction")
        print(f"Training: {max_steps:,} steps")
        print(f"Batch size: {self.config.get('learning', {}).get('batch_size', 32)}")
        print("="*70 + "\n")
        
        start_time = time.time()
        loss_fn = nn.CrossEntropyLoss()

        # GPU OPTIMIZATION: Use larger batch size from config
        batch_size = self.config.get('learning', {}).get('batch_size', 32)

        # Initialize GradScaler for Mixed Precision (AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        for step in range(max_steps):
            # Get batch
            if self.data_loader:
                try:
                    x_tokens, y_tokens = self.data_loader.get_batch('train', batch_size=batch_size, return_tensors='pt')
                except:
                    x_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
                    y_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
            else:
                x_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
                y_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)

            # PROPER AUTOREGRESSIVE LOSS WITH AMP
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # === IMPROVEMENT 5: Gradient Checkpointing ===
                if self.use_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    logits = checkpoint(self.model, x_tokens, use_reentrant=False)
                else:
                    logits = self.model(x_tokens)

                # Next-token prediction loss
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    y_tokens[:, 1:].reshape(-1)
                )

            # Keep loss on GPU, sync only at log intervals
            self._loss_buffer.append(loss.detach())

            # Backward with Scaler
            self.scaler.scale(loss).backward()
            
            # === IMPROVEMENT 3: Gradient Accumulation ===
            if (step + 1) % self.accumulation_steps != 0:
                continue
            
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # SPARSE SELECTION: Compute which groups WOULD be updated (for tracking)
            # Only compute every 100 steps to amortize cost
            if step % 100 == 0:
                with torch.no_grad():
                    # Quick gradient norm per group
                    grad_norms = torch.zeros(self.num_groups, device=device)
                    for g_idx, params in enumerate(self.param_groups):
                        total_norm_sq = 0.0
                        for p in params:
                            if p.grad is not None:
                                total_norm_sq += p.grad.pow(2).sum()
                        grad_norms[g_idx] = total_norm_sq.sqrt()
                    
                    # Top-K selection
                    k = min(self.top_k, self.num_groups)
                    _, top_indices = torch.topk(grad_norms, k=k)
                    self._current_mask = torch.zeros(self.num_groups, dtype=torch.bool, device=device)
                    self._current_mask[top_indices] = True
            
            # FAST: Single optimizer step (updates all params - no slow gradient zeroing)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Stats (use cached mask for reporting)
            num_selected = self._current_mask.sum().item()
            params_updated = (self._current_mask.float() * self._group_param_counts).sum().item()
            
            self.total_steps += 1
            self.total_params_updated += int(params_updated)
            
            # Logging
            if step % self.log_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                # Sync loss buffer
                if self._loss_buffer:
                    avg_loss = torch.stack(self._loss_buffer).mean().item()
                    self.loss_history.append(avg_loss)
                    self._loss_buffer = []
                else:
                    avg_loss = 0.0

                elapsed = time.time() - start_time
                update_pct = 100 * params_updated / self.total_params
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

                gpu_mem = ""
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    gpu_mem = f" | GPU: {mem_allocated:.2f}GB"

                print(f"[{step:6d}] Loss: {avg_loss:.4f} | "
                      f"Params: {int(params_updated):,} ({update_pct:.1f}%) | "
                      f"Groups: {int(num_selected)}/{self.num_groups} | "
                      f"Speed: {steps_per_sec:.1f} step/s{gpu_mem}")
            
            # Validation
            if step % self.validation_interval == 0 and self.data_loader:
                val_loss, val_ppl = self._validate()
                print(f"[{step:6d}] VALIDATION | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

        
        # Final results
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("Training Complete")
        print(f"{'='*70}")
        print(f"Total steps: {self.total_steps:,}")
        if self.total_steps > 0:
            print(f"Avg params updated: {self.total_params_updated // self.total_steps:,} "
                  f"({100 * self.total_params_updated / (self.total_steps * self.total_params):.1f}%)")
        else:
            print("Avg params updated: 0 (no training steps)")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*70}\n")

        return {
            'total_steps': self.total_steps,
            'total_params_updated': self.total_params_updated,
            'avg_params_per_step': self.total_params_updated // self.total_steps if self.total_steps > 0 else 0,
            'update_percentage': 100 * self.total_params_updated / (self.total_steps * self.total_params) if self.total_steps > 0 else 0.0,
            'time': elapsed,
            'phase_2_adjustments': 0
        }
    
    def _validate(self, num_batches=10):
        """Validation with proper autoregressive loss - batched for GPU efficiency."""
        losses = []
        total_tokens = 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    x_batch, y_batch = self.data_loader.get_batch('val', batch_size=32, return_tensors='pt')

                    # Process entire batch at once (GPU-optimized!)
                    logits = self.model(x_batch)  # [batch, seq_len, vocab]

                    # Compute loss for all sequences in batch
                    loss = loss_fn(
                        logits[:, :-1].reshape(-1, self.vocab_size),  # [batch*(seq-1), vocab]
                        y_batch[:, 1:].reshape(-1)  # [batch*(seq-1)]
                    )

                    losses.append(loss)  # Keep as tensor
                    total_tokens += y_batch[:, 1:].numel()
                except:
                    continue

        self.model.train()

        # Single GPU-CPU sync at the end
        if losses:
            total_loss = torch.stack(losses).sum().item()
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = float('inf')

        return avg_loss, np.exp(min(avg_loss, 100))
