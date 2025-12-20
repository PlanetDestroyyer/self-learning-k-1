#!/usr/bin/env python3
"""
Train K-1 on Dataset 3: Third domain (continuing from Dataset 2)
Final continual learning test
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

# Load Dataset 3
print("="*70)
print("DATASET 3: Final Continual Learning Test")
print("="*70)
data_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)

print(f"Train samples: {len(data_loader.train_data):,}\n")

# Create trainer
trainer = HybridK1Trainer(config, data_loader=data_loader)

# Load from Dataset 2
load_path = 'models/k1_dataset2.pt'
if os.path.exists(load_path):
    print(f"✓ Loading model from {load_path}")
    checkpoint = torch.load(load_path)
    trainer.embedding.load_state_dict(checkpoint['embedding'])
    trainer.network.load_state_dict(checkpoint['network'])
    trainer.output_proj.load_state_dict(checkpoint['output_proj'])
    print("✓ Loaded! Final training phase...\n")
else:
    print(f"⚠ No checkpoint found, training from scratch\n")

# Train
max_steps = min(10000, len(data_loader.train_data))
print(f"Training for {max_steps:,} steps...\n")

results = trainer.train(max_steps=max_steps)

# Save final model
save_path = 'models/k1_final.pt'
torch.save({
    'embedding': trainer.embedding.state_dict(),
    'network': trainer.network.state_dict(),
    'output_proj': trainer.output_proj.state_dict(),
    'config': config,
    'results': results,
    'vocab_size': trainer.vocab_size
}, save_path)

print(f"\n✓ Final model saved to: {save_path}")
print(f"✓ Training time: {results['time']:.1f}s")
print("\n" + "="*70)
print("CONTINUAL LEARNING TEST COMPLETE")
print("="*70)
print("Now test with generate_k1.py to see if model remembers all 3 datasets!")
