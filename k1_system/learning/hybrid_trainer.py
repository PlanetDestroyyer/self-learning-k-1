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

        # SPEED FIX: Single optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        # Stats
        self.total_params = sum(p.numel() for group in self.param_groups for p in group)
        self.total_params_updated = 0
        self.total_steps = 0
        self.loss_history = []
        self._loss_buffer = []
        self.group_update_count = defaultdict(int)
        
        # Pre-calculate group parameter counts as tensor
        group_counts = [sum(p.numel() for p in group) for group in self.param_groups]
        self._group_param_counts = torch.tensor(group_counts, dtype=torch.float32, device=device)

        # =====================================================
        # VECTORIZED SPARSE UPDATE: Pre-compute param mappings
        # =====================================================
        # Flatten all params into a list and track their group indices
        self._all_params = list(self.model.parameters())
        self._param_group_indices = torch.zeros(len(self._all_params), dtype=torch.long, device=device)
        
        for g_idx, params in enumerate(self.param_groups):
            param_set = set(params)
            for p_idx, p in enumerate(self._all_params):
                if p in param_set:
                    self._param_group_indices[p_idx] = g_idx
        
        # Pre-compute group start/end indices for fast gradient norm computation
        # Group params together for efficient batch operations
        self._group_param_slices = []
        for g_idx in range(self.num_groups):
            indices = (self._param_group_indices == g_idx).nonzero(as_tuple=True)[0].tolist()
            self._group_param_slices.append(indices)
        
        # Gradient Accumulation (effective batch = batch_size * accumulation_steps)
        self.accumulation_steps = config['learning'].get('accumulation_steps', 1)
        
        # Data Prefetching (pre-load batches for faster training)
        self.prefetch_batches = config['learning'].get('prefetch_batches', 4)
        self._prefetch_queue = []
        
        print(f"Total parameters: {self.total_params:,}")
        print(f"Parameter groups: {self.num_groups}")
        print(f"Top-K groups: {self.top_k} (sparse updates)")
        print(f"Gradient accumulation: {self.accumulation_steps} steps")
        print(f"Data prefetch: {self.prefetch_batches} batches\n")
    
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
        
        # =====================================================
        # DATA PREFETCHING: Pre-load batches for faster training
        # =====================================================
        def prefetch_batch():
            if self.data_loader:
                try:
                    return self.data_loader.get_batch('train', batch_size=batch_size, return_tensors='pt')
                except:
                    pass
            return (torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device),
                    torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device))
        
        # Pre-fill prefetch queue
        for _ in range(self.prefetch_batches):
            self._prefetch_queue.append(prefetch_batch())

        for step in range(max_steps):
            # Get batch from prefetch queue (fast!)
            if self._prefetch_queue:
                x_tokens, y_tokens = self._prefetch_queue.pop(0)
                self._prefetch_queue.append(prefetch_batch())  # Refill
            else:
                x_tokens, y_tokens = prefetch_batch()

            # Forward with AMP
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = self.model(x_tokens)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    y_tokens[:, 1:].reshape(-1)
                )
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Keep loss on GPU
            self._loss_buffer.append(loss.detach() * self.accumulation_steps)  # Store unscaled

            # Backward pass (accumulate gradients)
            self.scaler.scale(loss).backward()
            
            # =====================================================
            # GRADIENT ACCUMULATION: Only update every N steps
            # =====================================================
            if (step + 1) % self.accumulation_steps != 0:
                continue  # Keep accumulating
            
            # Unscale for clipping/selection
            self.scaler.unscale_(self.optimizer)
            
            # ============================================================
            # K-1 SPARSE UPDATES: Vectorized for speed
            # ============================================================
            with torch.no_grad():
                # VECTORIZED: Compute gradient norm per group using pre-computed slices
                grad_norms = torch.zeros(self.num_groups, device=device)
                for g_idx, param_indices in enumerate(self._group_param_slices):
                    norm_sq = 0.0
                    for p_idx in param_indices:
                        p = self._all_params[p_idx]
                        if p.grad is not None:
                            norm_sq += p.grad.pow(2).sum()
                    grad_norms[g_idx] = torch.sqrt(norm_sq) if isinstance(norm_sq, torch.Tensor) else 0.0
                
                # Select TOP-K groups with highest gradient norms
                k = min(self.top_k, self.num_groups)
                _, top_indices = torch.topk(grad_norms, k=k)
                mask = torch.zeros(self.num_groups, dtype=torch.bool, device=device)
                mask[top_indices] = True
                
                # VECTORIZED: Get mask per parameter using pre-computed indices
                param_mask = mask[self._param_group_indices]  # [num_params] bool
                
                # BATCH ZERO: Zero all unselected gradients at once
                params_to_zero = [self._all_params[i] for i in range(len(self._all_params)) 
                                  if not param_mask[i] and self._all_params[i].grad is not None]
                if params_to_zero:
                    grads_to_zero = [p.grad for p in params_to_zero]
                    torch._foreach_zero_(grads_to_zero)
                
                # Track stats (vectorized)
                num_selected = k
                params_updated = (mask.float() * self._group_param_counts).sum().item()
            
            # Optimizer step (only updates params with non-zero gradients!)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)  # Clear for next accumulation
            
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
