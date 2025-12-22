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
    
    # Get dynamic scales from model (or fallback)
    dynamic_scales = getattr(model, '_dynamic_scales', {})
    
    # Role names for 4-level tree: Root → Node → Agent → Sub-Agent (Culprit)
    role_names = ["Root", "Node", "Agent", "Sub-Agent"]
    icons = ["✓", "⚠️", "⚠️", "🚨"]
    
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Step {step:5d} | Loss: {loss:.4f}")
        print(f"{'─' * 60}")
        print("Hierarchical Error Attribution (DYNAMIC SCALING):")
        print()
        
        for i, node_idx in enumerate(path):
            icon = icons[min(i, len(icons) - 1)]
            role = role_names[min(i, len(role_names) - 1)]
            
            # Get dynamic scale (or fallback to fixed)
            scale = dynamic_scales.get(node_idx, 0.1) * 100
            grad = grads[i]
            
            indent = "  " * i
            culprit_marker = " ← CULPRIT!" if i == len(path) - 1 else ""
            
            print(f"{indent}{icon} {role} (Node {node_idx}): grad={grad:.3f}, update={scale:.1f}%{culprit_marker}")
        
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


def print_specialization_analysis(model, loader, top_n=5):
    """Print which tokens each node specializes in."""
    print("\n" + "=" * 70)
    print("NODE SPECIALIZATION ANALYSIS")
    print("=" * 70)
    print("\nWhat does each node 'specialize' in? (tokens it handles most)")
    
    # Only show leaf nodes with data
    for node in model.all_nodes:
        if node.is_leaf and node.token_counts:
            top_tokens = node.get_top_tokens(top_n)
            if top_tokens:
                print(f"\n  Node {node.node_id} (Leaf):")
                print(f"    Updates: {node.update_count}, Unique tokens: {len(node.token_counts)}")
                print(f"    Top tokens handled:")
                for tid, count in top_tokens:
                    word = loader.decode([tid]) if hasattr(loader, 'decode') else f"[{tid}]"
                    print(f"      '{word}': {count} times")


def run_demo(steps=5000, log_interval=500, verbose_interval=1000):
    """Run the interpretability demo with 41-node tree."""
    
    print_header()
    
    # Load data
    print("Loading WikiText dataset...")
    loader = DataLoader('wikitext', seq_length=32)
    print(f"  Vocabulary: {loader.vocab_size} words")
    print(f"  Training sequences: {len(loader.train_data)}")
    
    # Create model with DEEPER TREE (41 nodes as per README)
    print("\nCreating K-1 Hierarchical Tree (41 nodes)...")
    model = HierarchicalTree(
        vocab_size=loader.vocab_size,
        embed_dim=128,
        ff_dim=256,
        num_heads=4,
        tree_depth=4,              # Depth 4: Root → Nodes → Agents → Sub-Agents
        branching_factor=[4, 3, 2]  # 4 Nodes, 3 Agents each, 2 Sub-Agents each
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Total nodes: {len(model.all_nodes)} (was 13, now 41 for finer attribution)")
    print(f"  Tree structure: 1 Root + 4 Nodes + 12 Agents + 24 Sub-Agents")
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
        
        # Track updates and specialization
        track_node_updates(model)
        
        # Record culprit and its tokens (for specialization tracking)
        if hasattr(model, '_error_path'):
            culprit_idx = model._error_path[-1]
            culprit_history.append(culprit_idx)
            
            # Record which tokens this culprit node handled
            culprit_node = model.all_nodes[culprit_idx]
            culprit_node.record_tokens(y)  # Track target tokens for this error
        
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
    for node_idx, count in culprit_counts.most_common(10):  # Top 10
        share = count / len(culprit_history) * 100
        print(f"Node {node_idx:<4} {count:<15} {share:>5.1f}%")
    
    # NEW: Specialization analysis
    print_specialization_analysis(model, loader, top_n=5)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("1. K-1 traces errors to specific nodes (INTERPRETABILITY)")
    print("2. Deeper tree (41 nodes) = finer-grained attribution")
    print("3. Each node develops specialization over time")
    print("=" * 70)


if __name__ == '__main__':
    run_demo(steps=5000, log_interval=500, verbose_interval=1000)

