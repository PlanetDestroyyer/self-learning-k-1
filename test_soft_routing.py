#!/usr/bin/env python3
"""
Quick Test for K-1 Soft Routing Fix
Designed for Google Colab - standalone and fast
"""

import sys
import os
import json
import torch
import numpy as np

print("="*70)
print("K-1 SOFT ROUTING TEST")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*70 + "\n")

# Import K-1 modules
try:
    from k1_system.learning import HybridK1Trainer
    from data.loader import DataLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you're in the project directory:")
    print("  cd /content/self-learning-k-1  # (on Colab)")
    sys.exit(1)

# Load config
print("Loading configuration...")
with open('k1_system/config/config_phase1.json', 'r') as f:
    config = json.load(f)

# Load data (small subset for quick test)
print("Loading WikiText-2 dataset...")
data_loader = DataLoader(
    dataset_name='wikitext',
    vocab_size=10000,
    seq_length=64
)

config['model']['vocab_size'] = data_loader.get_vocab_size()

print(f"✓ Vocabulary size: {data_loader.get_vocab_size()}")
print(f"✓ Train samples: {len(data_loader.train_data)}")
print()

# Create trainer
print("Creating K-1 trainer with SOFT ROUTING...")
trainer = HybridK1Trainer(config, data_loader=data_loader)

print("\n" + "="*70)
print("RUNNING TEST: 1,000 steps (~5 minutes)")
print("="*70)
print("What we're testing:")
print("  - Soft routing should activate 5-10+ agents (not just 2-3)")
print("  - This should increase update % from 4.3% to 30-50%")
print("  - If it works, perplexity should drop faster")
print("="*70 + "\n")

# Train for 1000 steps
results = trainer.train(data=None, max_steps=1000)

# Analyze results
print("\n" + "="*70)
print("TEST RESULTS ANALYSIS")
print("="*70)

update_pct = results.get('update_percentage', 0)
print(f"Parameter Update %: {update_pct:.1f}%")
print()

# Verdict
if update_pct > 30:
    print("✅ SUCCESS: Soft routing is working!")
    print("   Update % jumped from 4.3% to {:.1f}%".format(update_pct))
    print()
    print("NEXT STEPS:")
    print("  1. Run full training (57,708 steps)")
    print("  2. Expect perplexity to drop significantly")
    print("  3. Test continual learning experiments")
    print()
    print("TO RUN FULL TRAINING:")
    print("  python3 compare_baseline_vs_k1.py")
    
elif update_pct > 15:
    print("⚠️  PARTIAL SUCCESS: Some improvement")
    print("   Update % improved from 4.3% to {:.1f}%".format(update_pct))
    print("   But still not enough agents activated")
    print()
    print("NEXT STEPS:")
    print("  - Increase soft routing top_k parameter")
    print("  - Or implement 'activate all agents' approach")
    
elif update_pct > 8:
    print("⚠️  MINIMAL IMPROVEMENT")
    print("   Update % barely changed: 4.3% → {:.1f}%".format(update_pct))
    print()
    print("NEXT STEPS:")
    print("  - Soft routing may not be activating enough paths")
    print("  - Need to implement 'activate all agents' approach")
    
else:
    print("❌ FAILED: Soft routing didn't help")
    print("   Update % still at {:.1f}% (expected 30-50%)".format(update_pct))
    print()
    print("NEXT STEPS:")
    print("  - Debug soft routing implementation")
    print("  - Check if forward_pass.forward() is using 'soft' mode")
    print("  - May need to activate ALL agents explicitly")

print("="*70)

# Save results
results_file = "test_soft_routing_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'update_percentage': update_pct,
        'total_steps': results.get('total_steps', 0),
        'success': update_pct > 30
    }, f, indent=2)

print(f"\nResults saved to: {results_file}")
print("\nTest complete!")
