#!/usr/bin/env python3
"""
Train K-1 on Dataset 2: Python Code (continuing from WikiText)
Tests continual learning - does K-1 forget WikiText after learning code?
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

# Load Dataset 2: Python Code
print("="*70)
print("DATASET 2: Python Code (Domain Shift from WikiText)")
print("="*70)
print("Loading Python code dataset...")

# Use code_search_net Python subset
data_loader = DataLoader(dataset_name='code_python', vocab_size=10000, seq_length=64)

print(f"Train samples: {len(data_loader.train_data):,}\n")

# Create trainer
trainer = HybridK1Trainer(config, data_loader=data_loader)

# Load previous model weights (continual learning!)
load_path = 'models/k1_dataset1.pt'
if os.path.exists(load_path):
    print(f"✓ Loading pre-trained model from {load_path}")
    checkpoint = torch.load(load_path)
    trainer.model.load_state_dict(checkpoint['model'])
    print("✓ Loaded! Continuing training on CODE domain...\n")
else:
    print(f"⚠ No checkpoint found at {load_path}, training from scratch\n")

# Train on Python code
max_steps = min(10000, len(data_loader.train_data))
print(f"Training for {max_steps:,} steps...\n")

results = trainer.train(max_steps=max_steps)

# Save updated model
save_path = 'models/k1_dataset2.pt'
torch.save({
    'model': trainer.model.state_dict(),
    'config': config,
    'results': results,
    'vocab_size': trainer.vocab_size
}, save_path)

print(f"\n✓ Model saved to: {save_path}")
print(f"✓ Training time: {results['time']:.1f}s")
print("\nNOTE: Model learned to handle Python CODE after WikiText")
print("Run generate_k1.py with WikiText prompts to test if it forgot!")
