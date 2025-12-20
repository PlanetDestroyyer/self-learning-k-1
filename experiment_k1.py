#!/usr/bin/env python3
"""
K-1 Continual Learning Experiment

Tests the K-1 hierarchical system on 3 datasets sequentially:
1. WikiText-2 (general English)
2. Code (Python)
3. Scientific (ArXiv abstracts)

This tests whether K-1 can learn new domains without forgetting old ones.
"""

import sys
import json
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader
from k1_system.core.hierarchical_tree import HierarchicalK1Trainer


def train_on_dataset(trainer, data_loader, dataset_name, max_steps=5000):
    """Train on a single dataset."""
    print(f"\n{'='*70}")
    print(f"TRAINING ON: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Update trainer's data loader
    trainer.data_loader = data_loader
    trainer.vocab_size = data_loader.get_vocab_size()
    
    return trainer.train(max_steps=max_steps)


def evaluate_on_dataset(trainer, data_loader, dataset_name, num_batches=50):
    """Evaluate on a dataset without updating weights."""
    print(f"\nEvaluating on {dataset_name}...")
    
    trainer.model.eval()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                x, y = data_loader.get_batch('val', batch_size=32, return_tensors='pt')
                logits, _ = trainer.model(x)
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
    print("K-1 CONTINUAL LEARNING EXPERIMENT")
    print("=" * 70)
    print("Testing: WikiText → Code → Scientific")
    print("Goal: Learn new domains without forgetting old ones")
    print("=" * 70)
    
    # Load config
    config_path = Path(__file__).parent / 'k1_system' / 'config' / 'config_phase1.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Config for experiment
    config['model']['tree_depth'] = 3
    config['model']['branching_factor'] = 3
    config['learning']['top_k'] = 5
    config['learning']['log_interval'] = 5000  # Log every 5000 steps
    
    # Run for 1 epoch per dataset (steps = num_train_samples)
    
    # Load all datasets
    print("\nLoading datasets...")
    wiki_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)
    
    # Use wiki vocabulary for all datasets (continual learning)
    code_loader = DataLoader(dataset_name='code_python', vocab_size=10000, seq_length=64,
                            shared_vocab=wiki_loader)
    sci_loader = DataLoader(dataset_name='scientific', vocab_size=10000, seq_length=64,
                           shared_vocab=wiki_loader)
    
    print(f"WikiText samples: ~{len(wiki_loader.train_data) if hasattr(wiki_loader, 'train_data') else 'N/A'}")
    
    # Create trainer with WikiText
    trainer = HierarchicalK1Trainer(config, wiki_loader)
    
    # Get epoch size (1 epoch = all training samples)
    wiki_epoch = len(wiki_loader.train_data[0]) if isinstance(wiki_loader.train_data, tuple) else 50000
    code_epoch = len(code_loader.train_data[0]) if isinstance(code_loader.train_data, tuple) else 5000
    sci_epoch = len(sci_loader.train_data[0]) if isinstance(sci_loader.train_data, tuple) else 5000
    
    print(f"\nTraining epochs: Wiki={wiki_epoch}, Code={code_epoch}, Sci={sci_epoch}")
    
    # Results storage
    results = {
        'after_wiki': {},
        'after_code': {},
        'after_scientific': {}
    }
    
    # ========== PHASE 1: Train on WikiText ==========
    train_on_dataset(trainer, wiki_loader, "WikiText", wiki_epoch)
    
    print("\n--- Evaluation after WikiText ---")
    results['after_wiki']['wiki'] = evaluate_on_dataset(trainer, wiki_loader, "WikiText")
    
    # ========== PHASE 2: Train on Code ==========
    train_on_dataset(trainer, code_loader, "Code (Python)", code_epoch)
    
    print("\n--- Evaluation after Code ---")
    results['after_code']['wiki'] = evaluate_on_dataset(trainer, wiki_loader, "WikiText")
    results['after_code']['code'] = evaluate_on_dataset(trainer, code_loader, "Code")
    
    # ========== PHASE 3: Train on Scientific ==========
    train_on_dataset(trainer, sci_loader, "Scientific", sci_epoch)
    
    print("\n--- Evaluation after Scientific ---")
    results['after_scientific']['wiki'] = evaluate_on_dataset(trainer, wiki_loader, "WikiText")
    results['after_scientific']['code'] = evaluate_on_dataset(trainer, code_loader, "Code")
    results['after_scientific']['sci'] = evaluate_on_dataset(trainer, sci_loader, "Scientific")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING RESULTS")
    print("=" * 70)
    
    print("\n📊 Perplexity Changes (lower = better):")
    print("-" * 50)
    print(f"{'Dataset':<15} {'After Wiki':<15} {'After Code':<15} {'After Sci':<15}")
    print("-" * 50)
    
    wiki_ppl_1 = results['after_wiki']['wiki'][1]
    wiki_ppl_2 = results['after_code']['wiki'][1]
    wiki_ppl_3 = results['after_scientific']['wiki'][1]
    
    code_ppl_2 = results['after_code']['code'][1]
    code_ppl_3 = results['after_scientific']['code'][1]
    
    sci_ppl_3 = results['after_scientific']['sci'][1]
    
    print(f"{'WikiText':<15} {wiki_ppl_1:<15.2f} {wiki_ppl_2:<15.2f} {wiki_ppl_3:<15.2f}")
    print(f"{'Code':<15} {'-':<15} {code_ppl_2:<15.2f} {code_ppl_3:<15.2f}")
    print(f"{'Scientific':<15} {'-':<15} {'-':<15} {sci_ppl_3:<15.2f}")
    
    # Forgetting analysis
    wiki_forgetting = (wiki_ppl_3 - wiki_ppl_1) / wiki_ppl_1 * 100
    code_forgetting = (code_ppl_3 - code_ppl_2) / code_ppl_2 * 100
    
    print("\n🧠 Forgetting Analysis:")
    print(f"  WikiText forgetting: {wiki_forgetting:+.1f}% (after learning Code + Sci)")
    print(f"  Code forgetting:     {code_forgetting:+.1f}% (after learning Sci)")
    
    if wiki_forgetting < 20 and code_forgetting < 20:
        print("\n✅ K-1 shows MINIMAL forgetting!")
    else:
        print("\n⚠️  Some forgetting observed (may need tuning)")
    
    # Save checkpoint
    checkpoint_path = Path(__file__).parent / 'checkpoints' / 'k1_continual.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'results': results,
        'config': config,
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved: {checkpoint_path}")


if __name__ == '__main__':
    main()
