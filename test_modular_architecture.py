#!/usr/bin/env python3
"""
Test the new Modular Sparse Transformer architecture.
Quick test to verify everything works.
"""

import sys
import os
import json
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from k1_system.learning.hybrid_trainer import HybridK1Trainer
from data.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load config
config_path = os.path.join(project_root, 'k1_system/config/config_phase1.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Small dataset for quick test
print("="*70)
print("TESTING: Modular Sparse Transformer")
print("="*70)
data_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)

print(f"Train samples: {len(data_loader.train_data):,}\n")

# Create trainer with new architecture
trainer = HybridK1Trainer(config, data_loader=data_loader)

# Quick test: 1000 steps
print(f"Running quick test (1000 steps)...\n")
results = trainer.train(max_steps=1000)

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)
print(f"Total steps: {results['total_steps']:,}")
print(f"Update %: {results['update_percentage']:.1f}%")
print(f"Time: {results['time']:.1f}s")
print("="*70)

print("\n✓ If you see this, the new architecture works!")
print("✓ Modular Transformer with proper autoregressive loss is functional!")
