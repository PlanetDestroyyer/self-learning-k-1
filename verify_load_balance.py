#!/usr/bin/env python3
"""
Verify Load Balancing Fix - Quick test to check node update distribution
"""
import sys
sys.path.insert(0, '/home/x/projects/self-learning-k-1')

import torch
from k1_system.core import HierarchicalTree
from k1_system.training import HierarchicalK1Trainer
from data.loader import DataLoader

def main():
    print("=" * 60)
    print("LOAD BALANCING VERIFICATION TEST")
    print("=" * 60)
    
    # Create small model
    loader = DataLoader('wikitext', seq_length=32)
    model = HierarchicalTree(
        vocab_size=loader.vocab_size,
        embed_dim=128,
        ff_dim=256,
        num_heads=4,
        tree_depth=3,
        branching_factor=[3, 3]
    )
    
    # Simple training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\nTraining for 500 steps...")
    model.train()
    
    for step in range(500):
        x, y = loader.get_batch('train', batch_size=16, return_tensors='pt')
        
        optimizer.zero_grad()
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits.view(-1, loader.vocab_size), y.view(-1))
        loss.backward()
        
        # Apply hierarchical step (the key function we're testing)
        model.fast_hierarchical_step(loss, step)
        
        optimizer.step()
        
        # Update node counts
        if hasattr(model, '_last_scales_indices'):
            for idx in model._last_scales_indices:
                node = model.all_nodes[idx]
                if not hasattr(node, 'update_count'):
                    node.update_count = 0
                node.update_count += 1
                node.last_updated_step = step
        
        if step % 100 == 0:
            print(f"  Step {step}: Loss={loss.item():.4f}")
    
    # Analyze node update distribution
    print("\n" + "=" * 60)
    print("NODE UPDATE DISTRIBUTION")
    print("=" * 60)
    
    update_counts = []
    for i, node in enumerate(model.all_nodes):
        count = getattr(node, 'update_count', 0)
        update_counts.append(count)
        node_type = "Root" if i == 0 else ("Leaf" if node.is_leaf else "Internal")
        print(f"  Node {i:2d} ({node_type:8s}): {count:5d} updates")
    
    total_updates = sum(update_counts)
    if total_updates > 0:
        # Calculate metrics
        percentages = [c / total_updates * 100 for c in update_counts]
        top3_pct = sum(sorted(percentages, reverse=True)[:3])
        max_pct = max(percentages)
        min_pct = min(percentages) if min(percentages) > 0 else 0
        
        # Gini coefficient
        n = len(update_counts)
        mean = sum(update_counts) / n
        gini = sum(abs(update_counts[i] - update_counts[j]) 
                   for i in range(n) for j in range(n)) / (2 * n * n * mean + 1e-8)
        
        print("\n" + "-" * 60)
        print("LOAD BALANCE METRICS:")
        print(f"  Total updates: {total_updates}")
        print(f"  Top 3 nodes share: {top3_pct:.1f}%")
        print(f"  Most used node: {max_pct:.1f}%")
        print(f"  Least used node: {min_pct:.1f}%")
        print(f"  Gini coefficient: {gini:.3f} (0=perfect balance, 1=complete imbalance)")
        
        # Check balance auxiliary loss
        if hasattr(model, '_load_balance_loss'):
            print(f"  Auxiliary load balance loss: {model._load_balance_loss:.4f}")
        
        print("-" * 60)
        if top3_pct < 50:
            print("✅ BALANCED: Top 3 nodes handle <50% of updates")
        elif top3_pct < 70:
            print("⚠️  MODERATE: Top 3 nodes handle {:.1f}% of updates".format(top3_pct))
        else:
            print("❌ IMBALANCED: Top 3 nodes handle {:.1f}% of updates".format(top3_pct))
    else:
        print("\n⚠️  No updates recorded")

if __name__ == '__main__':
    main()
