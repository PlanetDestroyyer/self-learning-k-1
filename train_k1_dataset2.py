#!/usr/bin/env python3
"""
Train K-1 on Dataset 2: Different domain (continuing from Dataset 1)
Tests continual learning - does K-1 forget Dataset 1?
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

# Load Dataset 2 (you can change this to any dataset)
print("="*70)
print("DATASET 2: Continuing Learning (New Domain)")
print("="*70)
print("Loading new domain data...")

# For now, use WikiText-2 again (you can replace with code/math/etc)
# To test with different dataset, install: pip install datasets
# Then use: dataset_name='openwebtext' or 'code_search_net'
data_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)

print(f"Train samples: {len(data_loader.train_data):,}\n")

# Create trainer
trainer = HybridK1Trainer(config, data_loader=data_loader)

# Load previous model weights (continual learning!)
load_path = 'models/k1_dataset1.pt'
if os.path.exists(load_path):
    print(f"✓ Loading pre-trained model from {load_path}")
    checkpoint = torch.load(load_path)
    trainer.embedding.load_state_dict(checkpoint['embedding'])
    trainer.network.load_state_dict(checkpoint['network'])
    trainer.output_proj.load_state_dict(checkpoint['output_proj'])
    print("✓ Loaded! Continuing training...\n")
else:
    print(f"⚠ No checkpoint found at {load_path}, training from scratch\n")

# Train on new dataset
max_steps = min(10000, len(data_loader.train_data))  # 10K steps
print(f"Training for {max_steps:,} steps...\n")

results = trainer.train(max_steps=max_steps)

# Save updated model
save_path = 'models/k1_dataset2.pt'
torch.save({
    'embedding': trainer.embedding.state_dict(),
    'network': trainer.network.state_dict(),
    'output_proj': trainer.output_proj.state_dict(),
    'config': config,
    'results': results,
    'vocab_size': trainer.vocab_size
}, save_path)

print(f"\n✓ Model saved to: {save_path}")
print(f"✓ Training time: {results['time']:.1f}s")
print("\nNOTE: To test continual learning, run generate_k1.py on Dataset 1 prompts")
print("to see if the model still remembers what it learned before!")
