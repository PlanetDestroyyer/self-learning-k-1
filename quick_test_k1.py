#!/usr/bin/env python3
"""
K-1 QUICK TEST - Short version for fast evaluation
Trains for 5K steps per dataset (~30 minutes total)
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader
from k1_system.training import HierarchicalK1Trainer


def evaluate_on_dataset(trainer, data_loader, dataset_name, num_batches=50):
    """Evaluate on a dataset."""
    print(f"\nEvaluating on {dataset_name}...")
    
    trainer.model.eval()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                x, y = data_loader.get_batch('val', batch_size=32)
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
    print("K-1 QUICK TEST - 5K steps per dataset")
    print("=" * 70)
    
    # Load config
    config_path = Path(__file__).parent / 'k1_system' / 'config' / 'config_phase1.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # FIXED CONFIG FOR QUICK TEST
    config['model']['tree_depth'] = 3
    config['model']['branching_factor'] = 3
    config['model']['embed_dim'] = 128
    config['learning']['top_k'] = 5
    config['learning']['batch_size'] = 64
    config['learning']['log_interval'] = 500
    config['learning']['learning_rate'] = 0.001
    
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
    
    # Create trainer
    trainer = HierarchicalK1Trainer(config, wiki_loader)
    
    # QUICK TEST: 5K steps per dataset (~10 min each)
    STEPS_PER_DATASET = 5000
    
    print(f"\n🚀 Quick test mode: {STEPS_PER_DATASET} steps per dataset")
    print(f"   Expected time: ~30 minutes total\n")
    
    results = {}
    
    # ========== TRAIN ON WIKITEXT ==========
    print("=" * 70)
    print("TRAINING ON: WIKITEXT")
    print("=" * 70)
    trainer.data_loader = wiki_loader
    trainer.train(max_steps=STEPS_PER_DATASET)
    
    # Show node update stats
    print("\n📊 Node Update Distribution (WikiText):")
    print("-" * 50)
    node_counts = {}
    for node in trainer._model_unwrapped.all_nodes:
        if hasattr(node, 'last_updated_step') and node.last_updated_step >= 0:
            node_id = f"Node_{node.node_id}" if hasattr(node, 'node_id') else f"Node_{id(node)}"
            node_counts[node_id] = node_counts.get(node_id, 0) + 1
    
    if node_counts:
        sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"{'Node':<20} {'Update Count':<15} {'% of total':<15}")
        print("-" * 50)
        total_updates = sum(node_counts.values())
        for node_id, count in sorted_nodes[:10]:  # Top 10
            pct = (count / total_updates * 100) if total_updates > 0 else 0
            print(f"{node_id:<20} {count:<15} {pct:>6.1f}%")
        
        # Check for concentration
        top_3_pct = sum(c for _, c in sorted_nodes[:3]) / total_updates * 100 if total_updates > 0 else 0
        if top_3_pct > 60:
            print(f"\n⚠️  Top 3 nodes handle {top_3_pct:.1f}% of updates (rich-get-richer!)")
        else:
            print(f"\n✅  Balanced: Top 3 nodes handle {top_3_pct:.1f}% of updates")
    
    print("\n--- Evaluation after WikiText ---")
    results['after_wiki'] = {}
    results['after_wiki']['wiki'] = evaluate_on_dataset(trainer, wiki_loader, "WikiText")
    
    # ========== TRAIN ON CODE ==========
    print("\n" + "=" * 70)
    print("TRAINING ON: CODE")
    print("=" * 70)
    trainer.data_loader = code_loader
    
    # Reset node counters to track Code updates separately
    for node in trainer._model_unwrapped.all_nodes:
        node.last_updated_step = -1
    
    trainer.train(max_steps=STEPS_PER_DATASET)
    
    # Show node update stats for Code
    print("\n📊 Node Update Distribution (Code):")
    print("-" * 50)
    node_counts_code = {}
    for node in trainer._model_unwrapped.all_nodes:
        if hasattr(node, 'last_updated_step') and node.last_updated_step >= 0:
            node_id = f"Node_{node.node_id}" if hasattr(node, 'node_id') else f"Node_{id(node)}"
            node_counts_code[node_id] = node_counts_code.get(node_id, 0) + 1
    
    if node_counts_code:
        sorted_nodes = sorted(node_counts_code.items(), key=lambda x: x[1], reverse=True)
        print(f"{'Node':<20} {'Update Count':<15} {'% of total':<15}")
        print("-" * 50)
        total_updates = sum(node_counts_code.values())
        for node_id, count in sorted_nodes[:10]:
            pct = (count / total_updates * 100) if total_updates > 0 else 0
            print(f"{node_id:<20} {count:<15} {pct:>6.1f}%")
        
        top_3_pct = sum(c for _, c in sorted_nodes[:3]) / total_updates * 100 if total_updates > 0 else 0
        if top_3_pct > 60:
            print(f"\n⚠️  Top 3 nodes handle {top_3_pct:.1f}% of updates (rich-get-richer!)")
        else:
            print(f"\n✅  Balanced: Top 3 nodes handle {top_3_pct:.1f}% of updates")
    
    print("\n--- Evaluation after Code ---")
    results['after_code'] = {}
    results['after_code']['wiki'] = evaluate_on_dataset(trainer, wiki_loader, "WikiText")
    results['after_code']['code'] = evaluate_on_dataset(trainer, code_loader, "Code")
    
    # ========== TRAIN ON SCIENTIFIC ==========
    print("\n" + "=" * 70)
    print("TRAINING ON: SCIENTIFIC")
    print("=" * 70)
    trainer.data_loader = sci_loader
    
    # Reset for Scientific
    for node in trainer._model_unwrapped.all_nodes:
        node.last_updated_step = -1
    
    trainer.train(max_steps=STEPS_PER_DATASET)
    
    # Show node update stats for Scientific
    print("\n📊 Node Update Distribution (Scientific):")
    print("-" * 50)
    node_counts_sci = {}
    for node in trainer._model_unwrapped.all_nodes:
        if hasattr(node, 'last_updated_step') and node.last_updated_step >= 0:
            node_id = f"Node_{node.node_id}" if hasattr(node, 'node_id') else f"Node_{id(node)}"
            node_counts_sci[node_id] = node_counts_sci.get(node_id, 0) + 1
    
    if node_counts_sci:
        sorted_nodes = sorted(node_counts_sci.items(), key=lambda x: x[1], reverse=True)
        print(f"{'Node':<20} {'Update Count':<15} {'% of total':<15}")
        print("-" * 50)
        total_updates = sum(node_counts_sci.values())
        for node_id, count in sorted_nodes[:10]:
            pct = (count / total_updates * 100) if total_updates > 0 else 0
            print(f"{node_id:<20} {count:<15} {pct:>6.1f}%")
        
        top_3_pct = sum(c for _, c in sorted_nodes[:3]) / total_updates * 100 if total_updates > 0 else 0
        if top_3_pct > 60:
            print(f"\n⚠️  Top 3 nodes handle {top_3_pct:.1f}% of updates (rich-get-richer!)")
        else:
            print(f"\n✅  Balanced: Top 3 nodes handle {top_3_pct:.1f}% of updates")
    
    print("\n--- Evaluation after Scientific ---")
    results['after_sci'] = {}
    results['after_sci']['wiki'] = evaluate_on_dataset(trainer, wiki_loader, "WikiText")
    results['after_sci']['code'] = evaluate_on_dataset(trainer, code_loader, "Code")
    results['after_sci']['sci'] = evaluate_on_dataset(trainer, sci_loader, "Scientific")
    
    # ========== RESULTS ==========
    print("\n" + "=" * 70)
    print("K-1 QUICK TEST RESULTS")
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
    
    if wiki_forgetting < 20 and code_forgetting < 20:
        print("\n✅ K-1 shows MINIMAL forgetting!")
    else:
        print("\n⚠️ Some forgetting observed")
    
    # Save
    checkpoint_path = Path(__file__).parent / 'checkpoints' / 'k1_quick_test.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'results': results,
        'config': config,
    }, checkpoint_path)
    
    print(f"\n✅ Checkpoint saved: {checkpoint_path}")


if __name__ == '__main__':
    main()
