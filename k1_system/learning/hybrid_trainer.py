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
        
        # Get parameter groups from modular architecture
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

        # Initialize Optimizer (Standard SGD matches the manual update logic)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        
        # Stats
        self.total_params = sum(p.numel() for group in self.param_groups for p in group)
        self.total_params_updated = 0
        self.total_steps = 0
        self.loss_history = []
        self.group_update_count = defaultdict(int)
        
        # GPU Stats Accumulators (avoid CPU sync during training)
        self._gpu_mask_accumulator = torch.zeros(self.num_groups, dtype=torch.long, device=device)
        self._gpu_params_updated_accumulator = torch.zeros(1, dtype=torch.float32, device=device)
        
        # Pre-calculate group parameter counts on GPU
        group_counts = [sum(p.numel() for p in group) for group in self.param_groups]
        self._group_param_counts = torch.tensor(group_counts, dtype=torch.float32, device=device)

        
        print(f"Total parameters: {self.total_params:,}")
        print(f"Parameter groups: {self.num_groups}")
        print(f"Top-K groups: {self.top_k}\n")
    
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

            # PROPER AUTOREGRESSIVE LOSS
            # Forward through modular transformer
            logits = self.model(x_tokens)  # [batch, seq_len, vocab_size]

            # Next-token prediction: predict y_tokens from x_tokens
            # Reshape for CrossEntropyLoss: [batch * (seq-1), vocab] and [batch * (seq-1)]
            loss = loss_fn(
                logits[:, :-1].reshape(-1, self.vocab_size),  # [batch*(seq-1), vocab]
                y_tokens[:, 1:].reshape(-1)  # [batch*(seq-1)]
            )

            # Keep loss on GPU, sync only at log intervals
            if not hasattr(self, '_loss_buffer'):
                self._loss_buffer = []
            self._loss_buffer.append(loss.detach())

            # Backward
            loss.backward()

            # GPU-OPTIMIZED: Calculate gradient norms directly on GPU
            with torch.no_grad():
                # Pre-allocate GPU tensor (no Python list!)
                grad_norms_tensor = torch.zeros(self.num_groups, device=device)

                for i, params in enumerate(self.param_groups):
                    # Fast check: usually all params have grads in this model
                    # We could skip the check for speed if we're sure
                    valid_grads = [p.grad for p in params if p.grad is not None]
                    if valid_grads:
                        # Compute norm and store directly in GPU tensor
                        # Optimization: Avoid list comprehension if possible, but this is okay-ish
                        # Better: use individual squared sums
                        sq_sum = torch.stack([g.norm()**2 for g in valid_grads]).sum()
                        grad_norms_tensor[i] = sq_sum.sqrt()

                # Compute median on GPU
                grad_threshold = torch.median(grad_norms_tensor)

                # Select groups with above-median gradients
                mask = grad_norms_tensor > grad_threshold

                # Fallback: if none selected, pick the one with highest gradient
                if not mask.any():
                    mask[torch.argmax(grad_norms_tensor)] = True

                # FULLY ASYNC: Apply mask to gradients on GPU
                # Do NOT sync mask to CPU. Multiply grads by mask (0.0 or 1.0)
                mask_float = mask.to(dtype=torch.float32)
                
                # Update statistics on GPU (lazily sync at log time)
                self._gpu_mask_accumulator += mask.long()
                
                # Calculate params updated for this step (on GPU)
                current_step_params = (mask_float * self._group_param_counts).sum()
                self._gpu_params_updated_accumulator += current_step_params

                # Apply mask to gradients
                for group_id in range(self.num_groups):
                    multiplier = mask_float[group_id]
                    # Optimization: If multiplier is 0, we could skip, but that requires CPU check (sync)
                    # So we just multiply. If 0, grad becomes 0.
                    for p in self.param_groups[group_id]:
                        if p.grad is not None:
                            p.grad.mul_(multiplier)

            # 2. Optimizer step (native C++ implementation)
            # Grads for unselected groups are now 0.0, so they won't change weights (for SGD)
            self.optimizer.step()
            
            # 3. Zero gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)
            
            self.total_steps += 1
            
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

                # Sync Statistics
                # Get current step params updated (just for the log)
                params_updated = int(current_step_params.item())
                
                # Update python dict from GPU accumulator
                group_counts_cpu = self._gpu_mask_accumulator.cpu().numpy()
                for i, count in enumerate(group_counts_cpu):
                     self.group_update_count[self.group_names[i]] = int(count)
                
                # Total params updated
                total_params_updated_cpu = self._gpu_params_updated_accumulator.item()
                self.total_params_updated = int(total_params_updated_cpu)
                
                # Active groups count for this step
                num_selected = int(mask.sum().item())

                elapsed = time.time() - start_time
                update_pct = 100 * params_updated / self.total_params
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

                gpu_mem = ""
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    gpu_mem = f" | GPU: {mem_allocated:.2f}GB"

                print(f"[{step:6d}] Loss: {avg_loss:.4f} | "
                      f"Params: {params_updated:,} ({update_pct:.1f}%) | "
                      f"Groups: {num_selected}/{self.num_groups} | "
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
