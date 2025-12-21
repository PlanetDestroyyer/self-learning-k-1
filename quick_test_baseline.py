#!/usr/bin/env python3
"""
BASELINE QUICK TEST - Short version for fast evaluation
Trains for 5K steps per dataset (~30 minutes total)
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader


class BaselineTransformer(nn.Module):
    """Simple baseline transformer for fair comparison."""
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        h = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        h = self.transformer(h)
        return self.output_proj(h)


def evaluate_on_dataset(model, data_loader, dataset_name, vocab_size, device, num_batches=50):
    """Evaluate model."""
    print(f"\nEvaluating on {dataset_name}...")
    
    model.eval()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                x, y = data_loader.get_batch('val', batch_size=32)
                logits = model(x)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, vocab_size),
                    y[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
            except:
                continue
    
    model.train()
    avg_loss = total_loss / num_batches
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 10000)
    
    print(f"  {dataset_name}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
    return avg_loss, perplexity


def train_baseline(model, data_loader, device, max_steps=5000, lr=0.001, log_interval=500):
    """Train baseline model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0.0
    
    for step in range(max_steps):
        x, y = data_loader.get_batch('train', batch_size=64)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(
            logits[:, :-1].reshape(-1, data_loader.get_vocab_size()),
            y[:, 1:].reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % log_interval == 0 and step > 0:
            avg_loss = total_loss / step
            print(f"[{step:6d}] Loss: {avg_loss:.4f}")
    
    print(f"Training complete: {max_steps} steps")


def main():
    print("=" * 70)
    print("BASELINE QUICK TEST - 5K steps per dataset")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except:
        pass
    
    # Load datasets with MATCHING seq_length=32
    print("\nLoading datasets (seq_length=32)...")
    wiki_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=32)
    code_loader = DataLoader(dataset_name='code_python', vocab_size=10000, seq_length=32,
                             shared_vocab=wiki_loader)
    sci_loader = DataLoader(dataset_name='scientific', vocab_size=10000, seq_length=32,
                           shared_vocab=wiki_loader)
    
    # Create baseline model
    model = BaselineTransformer(vocab_size=10000, embed_dim=128, num_heads=4, num_layers=3)
    model = model.to(device)
    
    print(f"\nBaseline model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # QUICK TEST: 5K steps per dataset
    STEPS_PER_DATASET = 5000
    
    print(f"\n🚀 Quick test mode: {STEPS_PER_DATASET} steps per dataset")
    print(f"   Expected time: ~30 minutes total\n")
    
    results = {}
    
    # ========== TRAIN ON WIKITEXT ==========
    print("=" * 70)
    print("TRAINING ON: WIKITEXT")
    print("=" * 70)
    train_baseline(model, wiki_loader, device, max_steps=STEPS_PER_DATASET)
    
    print("\n--- Evaluation after WikiText ---")
    results['after_wiki'] = {}
    results['after_wiki']['wiki'] = evaluate_on_dataset(model, wiki_loader, "WikiText", 10000, device)
    
    # ========== TRAIN ON CODE ==========
    print("\n" + "=" * 70)
    print("TRAINING ON: CODE")
    print("=" * 70)
    train_baseline(model, code_loader, device, max_steps=STEPS_PER_DATASET)
    
    print("\n--- Evaluation after Code ---")
    results['after_code'] = {}
    results['after_code']['wiki'] = evaluate_on_dataset(model, wiki_loader, "WikiText", 10000, device)
    results['after_code']['code'] = evaluate_on_dataset(model, code_loader, "Code", 10000, device)
    
    # ========== TRAIN ON SCIENTIFIC ==========
    print("\n" + "=" * 70)
    print("TRAINING ON: SCIENTIFIC")
    print("=" * 70)
    train_baseline(model, sci_loader, device, max_steps=STEPS_PER_DATASET)
    
    print("\n--- Evaluation after Scientific ---")
    results['after_sci'] = {}
    results['after_sci']['wiki'] = evaluate_on_dataset(model, wiki_loader, "WikiText", 10000, device)
    results['after_sci']['code'] = evaluate_on_dataset(model, code_loader, "Code", 10000, device)
    results['after_sci']['sci'] = evaluate_on_dataset(model, sci_loader, "Scientific", 10000, device)
    
    # ========== RESULTS ==========
    print("\n" + "=" * 70)
    print("BASELINE QUICK TEST RESULTS")
    print("=" * 70)
    
    print("\n📊 Perplexity Progression:")
    print("-" * 50)
    print(f"{'Dataset':<15} {'After Wiki':<15} {'After Code':<15} {'After Sci':<15}")
    print("-" * 50)
    
    wiki_ppl_1 = results['after_wiki']['wiki'][1]
    wiki_ppl_2 = results['after_code']['wiki'][1]
    wiki_ppl_3 = results['after_sci']['wiki'][1]
    
    code_ppl_2 = results['after_code']['code'][1]
    code_ppl_3 = results['after_sci']['code'][1]
    
    sci_ppl_3 = results['after_sci']['sci'][1]
    
    print(f"{'WikiText':<15} {wiki_ppl_1:<15.2f} {wiki_ppl_2:<15.2f} {wiki_ppl_3:<15.2f}")
    print(f"{'Code':<15} {'-':<15} {code_ppl_2:<15.2f} {code_ppl_3:<15.2f}")
    print(f"{'Scientific':<15} {'-':<15} {'-':<15} {sci_ppl_3:<15.2f}")
    
    # Forgetting analysis
    wiki_forgetting = (wiki_ppl_3 - wiki_ppl_1) / wiki_ppl_1 * 100
    code_forgetting = (code_ppl_3 - code_ppl_2) / code_ppl_2 * 100
    
    print("\n🧠 Forgetting Analysis:")
    print(f"  WikiText forgetting: {wiki_forgetting:+.1f}%")
    print(f"  Code forgetting:     {code_forgetting:+.1f}%")
    
    if wiki_forgetting > 50 or code_forgetting > 50:
        print("\n⚠️ SIGNIFICANT FORGETTING (expected for baseline)")
    else:
        print("\n✅ Minimal forgetting")
    
    # Save
    checkpoint_path = Path(__file__).parent / 'checkpoints' / 'baseline_quick_test.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
    }, checkpoint_path)
    
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")


if __name__ == '__main__':
    main()
