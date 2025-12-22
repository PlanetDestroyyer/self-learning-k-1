#!/usr/bin/env python3
"""
K-1 Interpretability Demo - Visual Error Attribution

Shows the core K-1 innovation: tracing prediction errors to specific nodes
in the hierarchical tree, providing full transparency into what's happening.
"""

import sys
sys.path.insert(0, '/home/x/projects/self-learning-k-1')

import torch
import torch.nn as nn
import time
from k1_system.core import HierarchicalTree
from data.loader import DataLoader


def print_header():
    print("=" * 70)
    print("K-1 INTERPRETABILITY DEMO: Hierarchical Error Attribution")
    print("=" * 70)
    print()
    print("Core Innovation: Instead of updating ALL weights blindly,")
    print("K-1 traces errors to specific nodes and updates only those.")
    print()


def print_error_attribution(model, step, loss, verbose=True):
    """Print beautiful error attribution visualization."""
    
    if not hasattr(model, '_error_path'):
        return
    
    path = model._error_path
    grads = model._path_gradients if hasattr(model, '_path_gradients') else [0] * len(path)
    
    # Icons for different roles
    icons = ["✓", "⚠️", "🚨"]
    roles = ["Root", "Node", "Culprit"]
    scales = [5, 15, 100]
    
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Step {step:5d} | Loss: {loss:.4f}")
        print(f"{'─' * 60}")
        print("Hierarchical Error Attribution:")
        print()
        
        for i, node_idx in enumerate(path):
            icon = icons[min(i, 2)]
            role = roles[min(i, 2)]
            scale = scales[min(i, 2)]
            grad = grads[i]
            
            indent = "  " * i
            culprit_marker = " ← CULPRIT!" if i == len(path) - 1 else ""
            
            print(f"{indent}{icon} {role} (Node {node_idx}): grad={grad:.3f}, update={scale}%{culprit_marker}")
        
        print()
        print(f"Updated: {len(path)}/{len(model.all_nodes)} nodes ({len(path)/len(model.all_nodes)*100:.0f}%)")
        print(f"Preserved: {len(model.all_nodes) - len(path)}/{len(model.all_nodes)} nodes ({(len(model.all_nodes) - len(path))/len(model.all_nodes)*100:.0f}%)")
        print(f"{'─' * 60}")
    
    return {
        'path': path,
        'gradients': grads,
        'culprit': path[-1]
    }


def track_node_updates(model):
    """Track which nodes get updated."""
    if hasattr(model, '_last_scales_indices'):
        for idx in model._last_scales_indices:
            node = model.all_nodes[idx]
            if not hasattr(node, 'update_count'):
                node.update_count = 0
            node.update_count += 1


def print_node_statistics(model):
    """Print statistics about which nodes got updated."""
    print("\n" + "=" * 70)
    print("NODE UPDATE STATISTICS")
    print("=" * 70)
    
    updates = []
    for i, node in enumerate(model.all_nodes):
        count = getattr(node, 'update_count', 0)
        updates.append((i, count, node.is_leaf if hasattr(node, 'is_leaf') else False))
    
    total = sum(u[1] for u in updates)
    
    print(f"\n{'Node':<10} {'Type':<10} {'Updates':<10} {'Share':<10}")
    print("-" * 40)
    
    for node_idx, count, is_leaf in sorted(updates, key=lambda x: -x[1]):
        node_type = "Leaf" if is_leaf else ("Root" if node_idx == 0 else "Internal")
        share = count / total * 100 if total > 0 else 0
        bar = "█" * int(share / 5)
        print(f"Node {node_idx:<4} {node_type:<10} {count:<10} {share:>5.1f}% {bar}")
    
    print("-" * 40)
    print(f"Total updates: {total}")


def run_demo(steps=5000, log_interval=500, verbose_interval=1000):
    """Run the interpretability demo."""
    
    print_header()
    
    # Load data
    print("Loading WikiText dataset...")
    loader = DataLoader('wikitext', seq_length=32)
    print(f"  Vocabulary: {loader.vocab_size} words")
    print(f"  Training sequences: {len(loader.train_data)}")
    
    # Create model
    print("\nCreating K-1 Hierarchical Tree...")
    model = HierarchicalTree(
        vocab_size=loader.vocab_size,
        embed_dim=128,
        ff_dim=256,
        num_heads=4,
        tree_depth=3,
        branching_factor=[3, 3]
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Total nodes: {len(model.all_nodes)}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'=' * 70}")
    print(f"Training for {steps} steps with error attribution...")
    print(f"{'=' * 70}")
    
    model.train()
    total_loss = 0.0
    start_time = time.time()
    culprit_history = []
    
    for step in range(steps):
        # Get batch
        x, y = loader.get_batch('train', 16, return_tensors='pt')
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits.view(-1, loader.vocab_size), y.view(-1))
        
        # Backward pass
        loss.backward()
        
        # K-1 Error Attribution
        model.fast_hierarchical_step(loss, step)
        
        # Track updates
        track_node_updates(model)
        
        # Record culprit
        if hasattr(model, '_error_path'):
            culprit_history.append(model._error_path[-1])
        
        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if step % log_interval == 0 and step > 0:
            avg_loss = total_loss / step
            elapsed = time.time() - start_time
            speed = step / elapsed
            print(f"[{step:5d}] Loss: {avg_loss:.4f} | Speed: {speed:.1f} step/s")
        
        # Verbose attribution output
        if step % verbose_interval == 0:
            print_error_attribution(model, step, loss.item(), verbose=True)
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\nTraining complete: {steps} steps in {elapsed:.1f}s ({steps/elapsed:.1f} step/s)")
    
    # Node statistics
    print_node_statistics(model)
    
    # Culprit analysis
    print("\n" + "=" * 70)
    print("CULPRIT ANALYSIS")
    print("=" * 70)
    
    from collections import Counter
    culprit_counts = Counter(culprit_history)
    print("\nWhich nodes were identified as 'culprit' most often?")
    print(f"\n{'Node':<10} {'Times Culprit':<15} {'Share':<10}")
    print("-" * 40)
    for node_idx, count in culprit_counts.most_common():
        share = count / len(culprit_history) * 100
        print(f"Node {node_idx:<4} {count:<15} {share:>5.1f}%")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAY: Unlike traditional backprop, K-1 tells you")
    print("EXACTLY which node caused each error. This is interpretability!")
    print("=" * 70)


if __name__ == '__main__':
    run_demo(steps=5000, log_interval=500, verbose_interval=1000)
