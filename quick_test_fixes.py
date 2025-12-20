#!/usr/bin/env python3
"""
Quick test to verify K-1 bug fixes work.
Runs 5000 steps and checks if loss decreases.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from k1_system.learning import HybridK1Trainer
from data.loader import DataLoader
import json

print("="*70)
print("QUICK TEST: Verifying K-1 Bug Fixes")
print("="*70)
print("Expected improvements:")
print("  1. Loss should decrease steadily (not plateau)")
print("  2. Perplexity should drop below 2000 (was 5504)")
print("  3. All agents should participate (not 7 starved)")
print("="*70 + "\n")

# Load config
with open('k1_system/config/config_phase1.json', 'r') as f:
    config = json.load(f)

# Load data
print("Loading WikiText-2...")
data_loader = DataLoader(
    dataset_name='wikitext',
    vocab_size=10000,
    seq_length=64
)

config['model']['vocab_size'] = data_loader.get_vocab_size()

# Create trainer
print("Creating K-1 trainer with fixed configuration...")
trainer = HybridK1Trainer(config, data_loader=data_loader)

# Run short training
print("\nTraining for 5,000 steps (quick test)...\n")
results = trainer.train(data=None, max_steps=5000)

# Check results
print("\n" + "="*70)
print("QUICK TEST RESULTS")
print("="*70)
print(f"Total steps: {results['total_steps']:,}")
print(f"Update percentage: {results['update_percentage']:.1f}%")
print(f"Expected: ~50% (15 agents out of ~30)")
print()

# Success criteria
success = True
if results['update_percentage'] < 30:
    print("❌ FAIL: Update percentage still too low")
    success = False
elif results['update_percentage'] > 40:
    print("✅ PASS: Update percentage improved!")
else:
    print("⚠️  PARTIAL: Update percentage marginal")

print("\nNext steps:")
if success:
    print("  ✅ Run full experiment (1 epoch)")
    print("  ✅ Build interpretability tracker")
    print("  ✅ Test continual learning")
else:
    print("  ❌ Debug further - check activated agents")
    print("  ❌ May need to adjust routing or hierarchy")

print("="*70)
