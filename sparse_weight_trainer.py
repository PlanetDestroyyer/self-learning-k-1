#!/usr/bin/env python3
"""
Sparse Weight Update Trainer - Only update weights causing the error

Core Idea: Instead of updating entire nodes, only update the specific weights
with the highest gradient magnitudes (top-K%).

This is a more surgical approach to sparse updates that should better preserve
knowledge from previous domains.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Iterator, Tuple, Optional
import time


class SparseWeightTrainer:
    """
    Trainer that only updates the top-K% of weights with highest gradients.
    
    Key difference from standard training:
    - Normal: Update ALL weights based on gradients
    - This: Only update weights with gradient magnitude in top-K%
    
    This should reduce forgetting by not modifying weights that aren't
    directly responsible for the current error.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        learning_rate: float = 3e-4,
        sparsity: float = 0.05,  # Only update top 5% of weights
        use_amp: bool = True,
        device: str = None
    ):
        """
        Args:
            model: The model to train
            train_loader: Iterator yielding (input, target) batches
            learning_rate: Learning rate
            sparsity: Fraction of weights to update (0.05 = top 5%)
            use_amp: Use automatic mixed precision
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.sparsity = sparsity
        self.use_amp = use_amp
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp and self.device == 'cuda' else None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.total_weights = sum(p.numel() for p in self.model.parameters())
        self.weights_to_update = int(self.total_weights * sparsity)
        
        print(f"Sparse Weight Trainer initialized:")
        print(f"  Total weights: {self.total_weights:,}")
        print(f"  Sparsity: {sparsity*100:.1f}% (updating {self.weights_to_update:,} weights per step)")
        print(f"  Device: {self.device}")
    
    def _apply_sparse_mask(self):
        """
        Zero out gradients except for top-K% by magnitude.
        
        This is the core of sparse weight updates - only the weights
        with the highest gradient magnitudes get updated.
        """
        # Collect all gradients into a flat tensor
        all_grads = []
        grad_shapes = []
        
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.view(-1))
                grad_shapes.append(param.grad.shape)
            else:
                all_grads.append(torch.zeros(param.numel(), device=self.device))
                grad_shapes.append(param.shape)
        
        flat_grads = torch.cat(all_grads)
        
        # Find threshold for top-K%
        k = max(1, int(len(flat_grads) * self.sparsity))
        threshold = flat_grads.abs().topk(k).values[-1]
        
        # Apply mask to each parameter's gradient
        idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                numel = param.numel()
                # Keep only gradients above threshold
                mask = param.grad.abs() >= threshold
                param.grad *= mask.float()
                idx += numel
        
        return k, threshold.item()
    
    def train(self, max_steps: int = 1000, log_interval: int = 100):
        """
        Train with sparse weight updates.
        
        Args:
            max_steps: Maximum training steps
            log_interval: Steps between logging
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"SPARSE WEIGHT TRAINING: Top {self.sparsity*100:.1f}% of gradients")
        print(f"{'='*70}")
        
        for step in range(max_steps):
            try:
                batch_x, batch_y = next(self.train_loader)
            except StopIteration:
                # Reset iterator
                self.train_loader = iter(self.train_loader)
                batch_x, batch_y = next(self.train_loader)
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp and self.scaler:
                with autocast():
                    output = self.model(batch_x)
                    logits = output[0] if isinstance(output, tuple) else output
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        batch_y.view(-1)
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Apply sparse gradient mask
                k, threshold = self._apply_sparse_mask()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(batch_x)
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    batch_y.view(-1)
                )
                
                loss.backward()
                
                # Apply sparse gradient mask
                k, threshold = self._apply_sparse_mask()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if step % log_interval == 0:
                avg_loss = total_loss / max(1, step + 1)
                elapsed = time.time() - start_time
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                
                print(f"[{step:5d}] Loss: {avg_loss:.4f} | "
                      f"Updated: {k:,}/{self.total_weights:,} weights | "
                      f"Speed: {speed:.1f} step/s")
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete: {max_steps} steps in {elapsed:.1f}s")
        
        return total_loss / max_steps
    
    def evaluate(self, data_loader, name: str = ""):
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for _ in range(100):  # Evaluate on 100 batches
                try:
                    batch_x, batch_y = next(data_loader)
                except StopIteration:
                    break
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x)
                logits = output[0] if isinstance(output, tuple) else output
                
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    batch_y.view(-1)
                )
                
                total_loss += loss.item() * batch_y.numel()
                total_tokens += batch_y.numel()
        
        avg_loss = total_loss / max(1, total_tokens)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"  {name}: Loss={avg_loss:.4f}, PPL={ppl:.2f}")
        
        self.model.train()
        return avg_loss, ppl


# =============================================================================
# Quick Test Script
# =============================================================================

def run_quick_test():
    """Run a quick test comparing sparse weight updates vs standard updates."""
    import sys
    sys.path.insert(0, '/home/x/projects/self-learning-k-1')
    
    from k1_system.core import HierarchicalTree
    from data.loader import DataLoader
    
    print("=" * 70)
    print("SPARSE WEIGHT UPDATE TEST")
    print("=" * 70)
    
    # Load data
    print("\nLoading WikiText dataset...")
    wiki_loader = DataLoader('wikitext', seq_length=32)
    
    print("\nLoading Code dataset...")
    code_loader = DataLoader('code_python', seq_length=32, shared_vocab=wiki_loader)
    
    # Create model
    model = HierarchicalTree(
        vocab_size=wiki_loader.vocab_size,
        embed_dim=128,
        ff_dim=256,
        num_heads=4,
        tree_depth=3,
        branching_factor=[3, 3]
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different sparsity levels
    for sparsity in [0.05, 0.10, 0.20]:
        print(f"\n{'='*70}")
        print(f"TESTING SPARSITY: {sparsity*100:.0f}% (updating top {sparsity*100:.0f}% of weights)")
        print(f"{'='*70}")
        
        # Fresh model for each test
        model = HierarchicalTree(
            vocab_size=wiki_loader.vocab_size,
            embed_dim=128,
            ff_dim=256,
            num_heads=4,
            tree_depth=3,
            branching_factor=[3, 3]
        )
        
        trainer = SparseWeightTrainer(
            model=model,
            train_loader=iter(lambda: wiki_loader.get_batch('train', 16, 'pt'), None),
            learning_rate=3e-4,
            sparsity=sparsity
        )
        
        # Train on WikiText
        print("\n--- Training on WikiText (1000 steps) ---")
        trainer.train_loader = iter(lambda: wiki_loader.get_batch('train', 16, 'pt'), None)
        trainer.train(max_steps=1000, log_interval=200)
        
        print("\nEvaluating after WikiText:")
        wiki_ppl_after_wiki = trainer.evaluate(
            iter(lambda: wiki_loader.get_batch('val', 16, 'pt'), None),
            "WikiText"
        )[1]
        
        # Train on Code
        print("\n--- Training on Code (1000 steps) ---")
        trainer.train_loader = iter(lambda: code_loader.get_batch('train', 16, 'pt'), None)
        trainer.train(max_steps=1000, log_interval=200)
        
        print("\nEvaluating after Code:")
        wiki_ppl_after_code = trainer.evaluate(
            iter(lambda: wiki_loader.get_batch('val', 16, 'pt'), None),
            "WikiText"
        )[1]
        code_ppl_after_code = trainer.evaluate(
            iter(lambda: code_loader.get_batch('val', 16, 'pt'), None),
            "Code"
        )[1]
        
        # Calculate forgetting
        forgetting = ((wiki_ppl_after_code - wiki_ppl_after_wiki) / wiki_ppl_after_wiki) * 100
        
        print(f"\n📊 Results for {sparsity*100:.0f}% sparsity:")
        print(f"  WikiText PPL: {wiki_ppl_after_wiki:.2f} → {wiki_ppl_after_code:.2f}")
        print(f"  Forgetting: {forgetting:+.1f}%")


if __name__ == '__main__':
    run_quick_test()
