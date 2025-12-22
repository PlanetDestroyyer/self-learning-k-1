#!/usr/bin/env python3
"""
K-1 Multi-Domain Specialization Demo

Tests whether nodes specialize by domain when trained on multiple datasets:
1. WikiText (English text)
2. Code (Python)
3. Scientific (ArXiv papers)

Key question: Do different nodes become "specialists" for different domains?
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
    print("K-1 MULTI-DOMAIN SPECIALIZATION DEMO")
    print("=" * 70)
    print()
    print("Question: Do nodes specialize by domain when trained on multiple datasets?")
    print()


def train_on_domain(model, loader, domain_name, optimizer, criterion, steps, device, batch_size=64, log_interval=5000):
    """Train model on a single domain, tracking which nodes handle it."""
    
    print(f"\n{'=' * 70}")
    print(f"TRAINING ON: {domain_name.upper()}")
    print(f"{'=' * 70}")
    
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for step in range(steps):
        x, y = loader.get_batch('train', batch_size, return_tensors='pt')
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits.view(-1, loader.vocab_size), y.view(-1))
        
        loss.backward()
        model.fast_hierarchical_step(loss, step)
        
        # Record domain for culprit node
        if hasattr(model, '_error_path'):
            culprit_idx = model._error_path[-1]
            culprit_node = model.all_nodes[culprit_idx]
            culprit_node.record_domain(domain_name)
            culprit_node.record_tokens(y)
            
            # Also record for parent nodes with reduced weight
            for node_idx in model._error_path[:-1]:
                model.all_nodes[node_idx].record_domain(domain_name)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % log_interval == 0 and step > 0:
            avg_loss = total_loss / step
            elapsed = time.time() - start_time
            speed = step / elapsed
            print(f"[{step:4d}] Loss: {avg_loss:.4f} | Speed: {speed:.1f} step/s")
    
    elapsed = time.time() - start_time
    print(f"\n{domain_name} training complete: {steps} steps in {elapsed:.1f}s")
    
    return total_loss / steps


def print_domain_specialization(model):
    """Print which nodes specialize in which domains."""
    
    print("\n" + "=" * 70)
    print("DOMAIN SPECIALIZATION ANALYSIS")
    print("=" * 70)
    print("\nWhich nodes became 'specialists' for each domain?")
    print()
    
    # Collect leaf nodes with domain data
    specialists = {'wikitext': [], 'code': [], 'scientific': []}
    
    for node in model.all_nodes:
        if node.domain_counts:
            primary = node.get_primary_domain()
            if primary[0] and primary[2] > 40:  # >40% confidence
                if primary[0] in specialists:
                    specialists[primary[0]].append({
                        'node_id': node.node_id,
                        'is_leaf': node.is_leaf,
                        'confidence': primary[2],
                        'distribution': node.get_domain_distribution()
                    })
    
    # Print by domain
    for domain in ['wikitext', 'code', 'scientific']:
        print(f"\n📚 {domain.upper()} Specialists:")
        print("-" * 50)
        
        domain_specialists = sorted(specialists.get(domain, []), key=lambda x: -x['confidence'])
        
        if not domain_specialists:
            print("  No specialized nodes found (all nodes handle multiple domains)")
        else:
            for spec in domain_specialists[:5]:  # Top 5
                node_type = "Leaf" if spec['is_leaf'] else "Internal"
                print(f"  Node {spec['node_id']} ({node_type}): {spec['confidence']:.1f}% {domain}")
                dist = spec['distribution']
                bar = " | ".join([f"{d}: {p:.0f}%" for d, p in sorted(dist.items())])
                print(f"    Distribution: {bar}")
    
    # Summary table
    print("\n" + "-" * 70)
    print("FULL NODE DOMAIN DISTRIBUTION:")
    print("-" * 70)
    print(f"{'Node':<8} {'Type':<10} {'WikiText':<12} {'Code':<12} {'Scientific':<12} {'Primary':<12}")
    print("-" * 70)
    
    for node in model.all_nodes:
        if node.domain_counts and node.is_leaf:
            dist = node.get_domain_distribution()
            wiki = dist.get('wikitext', 0)
            code = dist.get('code', 0)
            sci = dist.get('scientific', 0)
            primary = node.get_primary_domain()[0] or '-'
            
            # Color indicator
            if wiki > code and wiki > sci:
                indicator = "📖"
            elif code > wiki and code > sci:
                indicator = "💻"
            elif sci > wiki and sci > code:
                indicator = "🔬"
            else:
                indicator = "⚖️"
            
            print(f"Node {node.node_id:<3} {'Leaf':<10} {wiki:>6.1f}%      {code:>6.1f}%      {sci:>6.1f}%      {indicator} {primary}")


def run_multi_domain_demo(steps_per_domain=1000):
    """Run multi-domain training and analyze specialization."""
    
    print_header()
    
    # Load datasets
    print("Loading datasets...")
    
    print("\n1. Loading WikiText...")
    wiki_loader = DataLoader('wikitext', seq_length=32)
    
    print("\n2. Loading Code (Python)...")
    code_loader = DataLoader('code_python', seq_length=32, shared_vocab=wiki_loader)
    
    print("\n3. Loading Scientific (ArXiv)...")
    sci_loader = DataLoader('scientific', seq_length=32, shared_vocab=wiki_loader)
    
    # Create model
    print("\nCreating K-1 Hierarchical Tree (41 nodes)...")
    model = HierarchicalTree(
        vocab_size=wiki_loader.vocab_size,
        embed_dim=128,
        ff_dim=256,
        num_heads=4,
        tree_depth=4,
        branching_factor=[4, 3, 2]
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"  Device: {device}")
    print(f"  Total nodes: {len(model.all_nodes)}")
    print(f"  Leaf nodes: {sum(1 for n in model.all_nodes if n.is_leaf)}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining plan: {steps_per_domain} steps per domain × 3 domains")
    print(f"Expected time: ~{steps_per_domain * 3 / 6 / 60:.1f} minutes")
    
    # Train on each domain
    train_on_domain(model, wiki_loader, 'wikitext', optimizer, criterion, steps_per_domain, device)
    train_on_domain(model, code_loader, 'code', optimizer, criterion, steps_per_domain, device)
    train_on_domain(model, sci_loader, 'scientific', optimizer, criterion, steps_per_domain, device)
    
    # Analyze specialization
    print_domain_specialization(model)
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    
    # Count specialists per domain
    for domain in ['wikitext', 'code', 'scientific']:
        count = 0
        for node in model.all_nodes:
            if node.is_leaf and node.domain_counts:
                primary = node.get_primary_domain()
                if primary[0] == domain and primary[2] > 50:
                    count += 1
        print(f"  {domain.upper()} specialists (>50% confidence): {count} nodes")
    
    print()
    print("If nodes DON'T specialize: All domains are handled by all nodes")
    print("If nodes DO specialize: Different nodes prefer different domains")
    print("=" * 70)


if __name__ == '__main__':
    run_multi_domain_demo(steps_per_domain=10000)

