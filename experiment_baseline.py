#!/usr/bin/env python3
"""
Baseline Continual Learning Experiment

Tests traditional backprop (updates ALL weights) on 3 datasets:
1. WikiText-2 (general English)
2. Code (Python)
3. Scientific (ArXiv abstracts)

Compare with K-1 to show K-1's advantage in preventing forgetting.
"""

import sys
import json
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaselineTransformer(nn.Module):
    """Simple transformer baseline that updates ALL weights every step."""

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, ff_dim=256, max_seq_len=64):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # OPTIMIZATION: Pre-compute and cache causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Baseline Transformer: {total_params:,} parameters")
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        seq_len = x.size(1)
        h = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)

        # Use pre-computed causal mask (MUCH faster!)
        mask = self.causal_mask[:seq_len, :seq_len]

        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.output_norm(h)
        logits = self.output_proj(h)
        return logits


class BaselineTrainer:
    """Trainer that updates ALL weights every step (traditional backprop)."""
    
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        
        vocab_size = data_loader.get_vocab_size()
        self.vocab_size = vocab_size
        self.seq_length = config['model'].get('max_seq_len', 64)
        
        self.model = BaselineTransformer(
            vocab_size=vocab_size,
            embed_dim=config['model'].get('embed_dim', 128),
            num_heads=config['model'].get('num_heads', 4),
            num_layers=4,
            ff_dim=config['model'].get('hidden_dim', 256),
            max_seq_len=self.seq_length
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config['learning'].get('learning_rate', 0.001),
                                           weight_decay=0.01)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        self.log_interval = config['learning'].get('log_interval', 500)
    
    def train(self, max_steps=5000):
        print(f"Training Baseline (updates 100% of weights)...")

        loss_fn = nn.CrossEntropyLoss()
        batch_size = self.config['learning'].get('batch_size', 32)
        print(f"🔍 BATCH SIZE: {batch_size}")  # DEBUG
        start_time = time.time()
        last_log_time = start_time
        last_log_step = 0
        total_loss = 0.0
        running_loss = 0.0

        for step in range(max_steps):
            try:
                x, y = self.data_loader.get_batch('train', batch_size=batch_size, return_tensors='pt')
            except:
                x = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
                y = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = self.model(x)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    y[:, 1:].reshape(-1)
                )

            self.scaler.scale(loss).backward()

            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate loss WITHOUT GPU-CPU sync (CRITICAL for speed!)
            running_loss += loss.detach()

            if step % self.log_interval == 0 and step > 0:
                # Only sync GPU here, during logging
                current_time = time.time()
                elapsed_since_log = current_time - last_log_time
                steps_since_log = step - last_log_step
                recent_speed = steps_since_log / elapsed_since_log if elapsed_since_log > 0 else 0

                # Single GPU-CPU sync per log interval
                avg_loss = (running_loss / (step + 1)).item()
                total_elapsed = current_time - start_time
                overall_speed = (step + 1) / total_elapsed

                print(f"[{step:6d}] Loss: {avg_loss:.4f} | Speed: {recent_speed:.1f} step/s (avg: {overall_speed:.1f})")

                last_log_time = current_time
                last_log_step = step

        total_loss = running_loss.item() if isinstance(running_loss, torch.Tensor) else running_loss
        return {'loss': total_loss / max_steps, 'time': time.time() - start_time}


def evaluate(trainer, data_loader, dataset_name, num_batches=50):
    """Evaluate on a dataset."""
    trainer.model.eval()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                x, y = data_loader.get_batch('val', batch_size=32, return_tensors='pt')
                logits = trainer.model(x)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, trainer.vocab_size),
                    y[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
            except:
                continue
    
    trainer.model.train()
    avg_loss = total_loss / num_batches
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 10000)
    print(f"  {dataset_name}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
    return avg_loss, perplexity


def main():
    print("=" * 70)
    print("BASELINE CONTINUAL LEARNING EXPERIMENT")
    print("=" * 70)
    print("Testing: WikiText → Code → Scientific")
    print("Method: Traditional backprop (updates ALL weights)")
    print("=" * 70)
    
    # Load config
    config_path = Path(__file__).parent / 'k1_system' / 'config' / 'config_phase1.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # OPTIMIZATION: Increase batch size for T4 GPU speed
    # Try 256 first. If OOM, reduce to 128, then 64
    config['learning']['batch_size'] = 256  # Start with maximum (uses ~4GB VRAM)
    config['learning']['log_interval'] = 500  # Log more frequently to see speed

    # Enable PyTorch optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune for your GPU
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat32
    torch.backends.cudnn.allow_tf32 = True

    print(f"🔍 DEBUG: Batch size set to {config['learning']['batch_size']}")
    print(f"🔍 DEBUG: CuDNN benchmark: {torch.backends.cudnn.benchmark}")

    # Run for 1 epoch per dataset
    
    # Load datasets
    print("\nLoading datasets...")
    wiki_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)
    code_loader = DataLoader(dataset_name='code_python', vocab_size=10000, seq_length=64,
                            shared_vocab=wiki_loader)
    sci_loader = DataLoader(dataset_name='scientific', vocab_size=10000, seq_length=64,
                           shared_vocab=wiki_loader)
    
    # Create trainer
    trainer = BaselineTrainer(config, wiki_loader)
    
    # Get epoch size (1 epoch = all training samples)
    wiki_epoch = len(wiki_loader.train_data[0]) if isinstance(wiki_loader.train_data, tuple) else 50000
    code_epoch = len(code_loader.train_data[0]) if isinstance(code_loader.train_data, tuple) else 5000
    sci_epoch = len(sci_loader.train_data[0]) if isinstance(sci_loader.train_data, tuple) else 5000
    
    print(f"\nTraining epochs: Wiki={wiki_epoch}, Code={code_epoch}, Sci={sci_epoch}")
    
    results = {
        'after_wiki': {},
        'after_code': {},
        'after_scientific': {}
    }
    
    # ========== PHASE 1: Train on WikiText ==========
    print(f"\n{'='*70}")
    print("TRAINING ON: WIKITEXT")
    print(f"{'='*70}")
    trainer.train(max_steps=wiki_epoch)
    
    print("\n--- Evaluation after WikiText ---")
    results['after_wiki']['wiki'] = evaluate(trainer, wiki_loader, "WikiText")
    
    # ========== PHASE 2: Train on Code ==========
    print(f"\n{'='*70}")
    print("TRAINING ON: CODE (PYTHON)")
    print(f"{'='*70}")
    trainer.data_loader = code_loader
    trainer.train(max_steps=code_epoch)
    
    print("\n--- Evaluation after Code ---")
    results['after_code']['wiki'] = evaluate(trainer, wiki_loader, "WikiText")
    results['after_code']['code'] = evaluate(trainer, code_loader, "Code")
    
    # ========== PHASE 3: Train on Scientific ==========
    print(f"\n{'='*70}")
    print("TRAINING ON: SCIENTIFIC")
    print(f"{'='*70}")
    trainer.data_loader = sci_loader
    trainer.train(max_steps=sci_epoch)
    
    print("\n--- Evaluation after Scientific ---")
    results['after_scientific']['wiki'] = evaluate(trainer, wiki_loader, "WikiText")
    results['after_scientific']['code'] = evaluate(trainer, code_loader, "Code")
    results['after_scientific']['sci'] = evaluate(trainer, sci_loader, "Scientific")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("BASELINE CONTINUAL LEARNING RESULTS")
    print("=" * 70)
    
    print("\n📊 Perplexity Changes (lower = better):")
    print("-" * 50)
    
    wiki_ppl_1 = results['after_wiki']['wiki'][1]
    wiki_ppl_2 = results['after_code']['wiki'][1]
    wiki_ppl_3 = results['after_scientific']['wiki'][1]
    code_ppl_2 = results['after_code']['code'][1]
    code_ppl_3 = results['after_scientific']['code'][1]
    sci_ppl_3 = results['after_scientific']['sci'][1]
    
    print(f"{'Dataset':<15} {'After Wiki':<15} {'After Code':<15} {'After Sci':<15}")
    print("-" * 50)
    print(f"{'WikiText':<15} {wiki_ppl_1:<15.2f} {wiki_ppl_2:<15.2f} {wiki_ppl_3:<15.2f}")
    print(f"{'Code':<15} {'-':<15} {code_ppl_2:<15.2f} {code_ppl_3:<15.2f}")
    print(f"{'Scientific':<15} {'-':<15} {'-':<15} {sci_ppl_3:<15.2f}")
    
    # Forgetting analysis
    wiki_forgetting = (wiki_ppl_3 - wiki_ppl_1) / wiki_ppl_1 * 100
    code_forgetting = (code_ppl_3 - code_ppl_2) / code_ppl_2 * 100
    
    print("\n🧠 Forgetting Analysis:")
    print(f"  WikiText forgetting: {wiki_forgetting:+.1f}% (after learning Code + Sci)")
    print(f"  Code forgetting:     {code_forgetting:+.1f}% (after learning Sci)")
    
    if wiki_forgetting > 30 or code_forgetting > 30:
        print("\n⚠️  SIGNIFICANT FORGETTING DETECTED!")
        print("    This is expected for traditional backprop.")
        print("    Compare with K-1 which should show less forgetting.")
    
    # Save
    checkpoint_path = Path(__file__).parent / 'checkpoints' / 'baseline_continual.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'results': results,
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved: {checkpoint_path}")


if __name__ == '__main__':
    main()
