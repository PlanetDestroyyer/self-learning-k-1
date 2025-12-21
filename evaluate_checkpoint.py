#!/usr/bin/env python3
"""
Evaluate K-1 Checkpoint - Quick evaluation of saved model
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader
from k1_system.training import HierarchicalK1Trainer


def evaluate_on_dataset(trainer, data_loader, dataset_name, num_batches=50):
    """Evaluate on a dataset without updating weights."""
    print(f"\nEvaluating on {dataset_name}...")
    
    trainer.model.eval()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                x, y = data_loader.get_batch('val', batch_size=32)
                logits = trainer.model(x)  # Model returns logits only
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, trainer.vocab_size),
                    y[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
            except Exception as e:
                continue
    
    trainer.model.train()
    avg_loss = total_loss / num_batches
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 10000)
    
    print(f"  {dataset_name}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
    return avg_loss, perplexity


def main():
    print("=" * 70)
    print("K-1 CHECKPOINT EVALUATION")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint_path = Path(__file__).parent / 'checkpoints' / 'k1_continual.pt'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Run experiment_k1.py first to create the checkpoint!")
        return
    
    print(f"\n✅ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    print(f"✅ Config loaded: batch_size={config['learning']['batch_size']}")
    
    # Load datasets
    print("\nLoading datasets...")
    wiki_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)
    code_loader = DataLoader(dataset_name='code_python', vocab_size=10000, seq_length=64,
                             shared_vocab=wiki_loader)
    sci_loader = DataLoader(dataset_name='scientific', vocab_size=10000, seq_length=64,
                           shared_vocab=wiki_loader)
    
    # Create trainer and load model
    print("\nCreating trainer and loading model...")
    trainer = HierarchicalK1Trainer(config, wiki_loader)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✅ Model loaded successfully!")
    
    # Evaluate on all datasets
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    results = {}
    results['wiki'] = evaluate_on_dataset(trainer, wiki_loader, "WikiText")
    results['code'] = evaluate_on_dataset(trainer, code_loader, "Code")
    results['sci'] = evaluate_on_dataset(trainer, sci_loader, "Scientific")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Loss':<12} {'Perplexity':<12}")
    print("-" * 40)
    print(f"{'WikiText':<15} {results['wiki'][0]:<12.4f} {results['wiki'][1]:<12.2f}")
    print(f"{'Code':<15} {results['code'][0]:<12.4f} {results['code'][1]:<12.2f}")
    print(f"{'Scientific':<15} {results['sci'][0]:<12.4f} {results['sci'][1]:<12.2f}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
