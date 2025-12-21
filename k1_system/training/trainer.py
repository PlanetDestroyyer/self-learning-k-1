"""
HierarchicalK1Trainer: Trainer for the K-1 hierarchical system.

Key innovation: Path-based gradient updates.
- Compute gradients for entire tree
- Identify high-gradient paths (responsible for errors)
- Update those paths more, others less
"""

import time
import torch
import torch.nn as nn
from typing import Optional

from ..core.tree import HierarchicalTree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HierarchicalK1Trainer:
    """
    Trainer for hierarchical K-1 system.
    
    Uses path-based gradient updates where only nodes responsible
    for errors receive significant updates.
    
    Attributes:
        model: HierarchicalTree model
        optimizer: AdamW optimizer
        scaler: AMP gradient scaler
        lr: Learning rate
        top_k_nodes: Number of top nodes to update
        log_interval: Steps between logging
    """
    
    def __init__(self, config: dict, data_loader=None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary with 'model' and 'learning' sections
            data_loader: Optional data loader instance
        """
        self.config = config
        self.data_loader = data_loader
        
        # Extract config values
        vocab_size = data_loader.get_vocab_size() if data_loader else 10000
        embed_dim = config['model'].get('embed_dim', 128)
        ff_dim = config['model'].get('hidden_dim', 256)
        num_heads = config['model'].get('num_heads', 4)
        tree_depth = config['model'].get('tree_depth', 3)
        branching_factor = config['model'].get('branching_factor', 3)
        max_seq_len = config['model'].get('max_seq_len', 64)
        
        # Create hierarchical model
        self.model = HierarchicalTree(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            tree_depth=tree_depth,
            branching_factor=branching_factor,
            max_seq_len=max_seq_len
        ).to(device)
        
        # Store reference to unwrapped model (for accessing tree nodes)
        self._model_unwrapped = self.model
        
        # Multi-GPU support (Kaggle has 2 GPUs)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"🚀 Multi-GPU detected: {self.num_gpus} GPUs")
            self.model = nn.DataParallel(self.model)
            print(f"✅ DataParallel enabled (~{self.num_gpus * 0.9:.1f}x speedup)")
        
        # Compile model for speed (PyTorch 2.0+) - AFTER DataParallel
        if hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✅ torch.compile() enabled (20-40% faster)")
            except Exception as e:
                print(f"⚠️ torch.compile() failed: {e}")
        
        # Training settings
        self.lr = config['learning'].get('learning_rate', 0.001)
        self.top_k_nodes = config['learning'].get('top_k', 5)
        self.log_interval = config['learning'].get('log_interval', 100)
        
        # Optimizer with fused kernels (faster on GPU)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=0.01,
            fused=True if device.type == 'cuda' else False
        )

        # AMP scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        
        # Stats
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.vocab_size = vocab_size
        self.seq_length = max_seq_len
        
        print(f"Total parameters: {self.total_params:,}")
        print(f"Nodes in tree: {len(self._model_unwrapped.all_nodes)}")
        print(f"Top-K nodes to update: {self.top_k_nodes}")
    
    def train(self, max_steps: int = 1000) -> dict:
        """
        Train with path-based gradient updates.
        
        Args:
            max_steps: Maximum training steps
            
        Returns:
            Dict with 'loss' and 'time' keys
        """
        print("=" * 70)
        print("HIERARCHICAL K-1: Path-Based Gradient Updates")
        print("=" * 70)
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Tree structure: depth={self._model_unwrapped.tree_depth}, "
              f"branching={self._model_unwrapped.branching_factor}")
        print("=" * 70)
        
        loss_fn = nn.CrossEntropyLoss()
        batch_size = self.config['learning'].get('batch_size', 32)
        start_time = time.time()
        total_loss = 0.0
        
        for step in range(max_steps):
            # Get batch
            x, y = self._get_batch(batch_size)
            
            # Forward pass
            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = self.model(x)  # Only logits returned (DataParallel compatible)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    y[:, 1:].reshape(-1)
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Cache gradient norms ONCE (optimization) - use unwrapped model
            self._model_unwrapped.cache_all_gradient_norms()
            
            # Get loss value once (avoid multiple GPU syncs)
            loss_val = loss.item()

            # Hierarchical error attribution - use unwrapped model
            with torch.no_grad():
                responsible_path = self._model_unwrapped.find_responsible_path(
                    loss=loss_val,
                    current_step=step
                )
                scales = self._model_unwrapped.get_proportional_scales(responsible_path)
                self._model_unwrapped.apply_proportional_updates(scales)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Mark updated nodes for cooldown
            self._model_unwrapped.mark_nodes_updated(scales, step)

            # Accumulate loss
            total_loss += loss.detach()
            
            # Logging (only compute expensive things when logging)
            if step % self.log_interval == 0:
                # Get grad norms from cache (already computed)
                grad_norms = self._model_unwrapped._grad_norm_cache.copy()
                nodes_to_update = [nid for nid, scale in scales.items() if scale > 0]
                self._log_progress(
                    step, total_loss, start_time, 
                    responsible_path, scales, grad_norms, nodes_to_update
                )
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete: {max_steps} steps in {elapsed:.1f}s")
        
        return {'loss': (total_loss / max_steps).item(), 'time': elapsed}
    
    def _get_batch(self, batch_size: int):
        """Get a training batch."""
        if self.data_loader:
            try:
                return self.data_loader.get_batch(
                    'train', batch_size=batch_size, return_tensors='pt'
                )
            except Exception:
                pass
        
        # Fallback to random
        x = torch.randint(
            0, self.vocab_size, (batch_size, self.seq_length), device=device
        )
        y = torch.randint(
            0, self.vocab_size, (batch_size, self.seq_length), device=device
        )
        return x, y
    
    def _log_progress(
        self, step, total_loss, start_time, 
        responsible_path, scales, grad_norms, nodes_to_update
    ):
        """Log training progress."""
        avg_loss = (total_loss / (step + 1)).item()
        elapsed = time.time() - start_time
        speed = (step + 1) / elapsed if elapsed > 0 else 0

        print(f"\n[{step:6d}] Loss: {avg_loss:.4f} | Speed: {speed:.1f} step/s")
        print("─" * 60)
        print("Hierarchical Error Attribution:")
        self._model_unwrapped.print_responsibility_tree(grad_norms, scales)

        # Show responsible path
        path_str = " → ".join(
            f"Node{node.node_id}(r={resp:.2f})"
            for node, resp in responsible_path
        )
        print(f"\nError Path: {path_str}")

        # Summary
        num_updated = len(nodes_to_update)
        num_total = len(self._model_unwrapped.all_nodes)
        pct_updated = (num_updated / num_total * 100) if num_total > 0 else 0
        print(f"Updated: {num_updated}/{num_total} nodes ({pct_updated:.0f}%) | "
              f"Preserved: {num_total - num_updated} nodes ({100-pct_updated:.0f}%)")
        print("─" * 60)
