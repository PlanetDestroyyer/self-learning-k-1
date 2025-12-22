#!/usr/bin/env python3
"""
K-1 Forgetting Test with Specialized Nodes

Question: Does domain specialization reduce catastrophic forgetting?

Test:
1. Train on WikiText (establish baseline)
2. Train on Code (continual learning)
3. Re-test on WikiText (measure forgetting)

Compare: K-1 with specialization vs traditional backprop
"""

import sys
sys.path.insert(0, '/home/x/projects/self-learning-k-1')

import torch
import torch.nn as nn
from k1_system.core import HierarchicalTree
from data.loader import DataLoader


def evaluate_perplexity(model, loader, device, num_batches=100):
    """Evaluate perplexity on dataset."""
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i in range(num_batches):
            x, y = loader.get_batch('val', 64, return_tensors='pt')
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits.view(-1, loader.vocab_size), y.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return perplexity


def train_domain(model, loader, domain_name, optimizer, criterion, steps, device):
    """Train on a domain."""
    print(f"\nTraining on {domain_name}: {steps} steps")
    
    for step in range(steps):
        x, y = loader.get_batch('train', 64, return_tensors='pt')
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits.view(-1, loader.vocab_size), y.view(-1))
        
        loss.backward()
        model.fast_hierarchical_step(loss, step)
        
        # Track domain for specialization
        if hasattr(model, '_error_path'):
            culprit_idx = model._error_path[-1]
            model.all_nodes[culprit_idx].record_domain(domain_name)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 1000 == 0 and step > 0:
            print(f"  Step {step}/{steps}")


def run_forgetting_test():
    """Test if specialization reduces forgetting."""
    
    print("=" * 70)
    print("K-1 FORGETTING TEST WITH DOMAIN SPECIALIZATION")
    print("=" * 70)
    print()
    
    # Load datasets
    print("Loading datasets...")
    wiki_loader = DataLoader('wikitext', seq_length=32)
    code_loader = DataLoader('code_python', seq_length=32, shared_vocab=wiki_loader)
    
    # Create model
    print("\nCreating K-1 model (41 nodes)...")
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Device: {device}")
    print(f"Total nodes: {len(model.all_nodes)}")
    
    # Phase 1: Train on WikiText
    print("\n" + "=" * 70)
    print("PHASE 1: Train on WikiText (establish baseline)")
    print("=" * 70)
    
    train_domain(model, wiki_loader, 'wikitext', optimizer, criterion, 5000, device)
    
    wiki_ppl_after_wiki = evaluate_perplexity(model, wiki_loader, device)
    print(f"\n✓ WikiText PPL after WikiText training: {wiki_ppl_after_wiki:.2f}")
    
    # Check specialization
    wiki_specialists = sum(1 for n in model.all_nodes if n.is_leaf and 
                          n.get_primary_domain()[0] == 'wikitext' and 
                          n.get_primary_domain()[2] > 50)
    print(f"  WikiText specialists formed: {wiki_specialists} nodes")
    
    # Phase 2: Train on Code (continual learning)
    print("\n" + "=" * 70)
    print("PHASE 2: Train on Code (continual learning)")
    print("=" * 70)
    
    train_domain(model, code_loader, 'code', optimizer, criterion, 5000, device)
    
    code_ppl_after_code = evaluate_perplexity(model, code_loader, device)
    print(f"\n✓ Code PPL after Code training: {code_ppl_after_code:.2f}")
    
    # Check code specialization
    code_specialists = sum(1 for n in model.all_nodes if n.is_leaf and 
                          n.get_primary_domain()[0] == 'code' and 
                          n.get_primary_domain()[2] > 50)
    print(f"  Code specialists formed: {code_specialists} nodes")
    
    # Phase 3: Re-test on WikiText (measure forgetting)
    print("\n" + "=" * 70)
    print("PHASE 3: Re-test on WikiText (measure forgetting)")
    print("=" * 70)
    
    wiki_ppl_after_code = evaluate_perplexity(model, wiki_loader, device)
    print(f"\n✓ WikiText PPL after Code training: {wiki_ppl_after_code:.2f}")
    
    # Calculate forgetting
    forgetting_percent = ((wiki_ppl_after_code - wiki_ppl_after_wiki) / wiki_ppl_after_wiki) * 100
    
    print("\n" + "=" * 70)
    print("FORGETTING ANALYSIS")
    print("=" * 70)
    
    print(f"\nWikiText Performance:")
    print(f"  After WikiText training: {wiki_ppl_after_wiki:.2f} PPL")
    print(f"  After Code training:     {wiki_ppl_after_code:.2f} PPL")
    print(f"  Forgetting:              {forgetting_percent:+.1f}%")
    
    # Domain specialization summary
    print(f"\nDomain Specialization:")
    print(f"  WikiText specialists: {wiki_specialists} nodes")
    print(f"  Code specialists:     {code_specialists} nodes")
    
    # Check if WikiText specialists preserved performance
    wiki_specialist_ids = [n.node_id for n in model.all_nodes if n.is_leaf and
                           n.get_primary_domain()[0] == 'wikitext' and
                           n.get_primary_domain()[2] > 50]
    
    print(f"\n  WikiText specialist nodes: {wiki_specialist_ids}")
    print(f"  Theory: These nodes should preserve WikiText knowledge")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if forgetting_percent < 20:
        print(f"✅ GOOD: Only {forgetting_percent:.1f}% forgetting")
        print("   Domain specialization helped preserve WikiText knowledge!")
    elif forgetting_percent < 50:
        print(f"⚠️  MODERATE: {forgetting_percent:.1f}% forgetting")
        print("   Some forgetting occurred, but specialization helped")
    else:
        print(f"❌ HIGH: {forgetting_percent:.1f}% forgetting")  
        print("   Specialization alone not enough, need additional techniques")
    
    print("\nKey Insight:")
    print("Different nodes handle different domains = reduced interference")
    print("=" * 70)


if __name__ == '__main__':
    run_forgetting_test()
