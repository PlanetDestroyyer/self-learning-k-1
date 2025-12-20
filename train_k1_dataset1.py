#!/usr/bin/env python3
"""
Train K-1 on Dataset 1: WikiText-2
Saves model checkpoint for later use
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
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Load config
config_path = os.path.join(project_root, 'k1_system/config/config_phase1.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Load WikiText-2
print("="*70)
print("DATASET 1: WikiText-2 (General Text)")
print("="*70)
data_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)

# Get number of samples (handle both list and GPU tensor formats)
if isinstance(data_loader.train_data, tuple):
    num_samples = data_loader.train_data[0].shape[0]
else:
    num_samples = len(data_loader.train_data)

print(f"Train samples: {num_samples:,}")
print(f"Vocab size: {data_loader.get_vocab_size():,}\n")

# Train K-1
trainer = HybridK1Trainer(config, data_loader=data_loader)
max_steps = num_samples  # 1 epoch

print(f"Training for 1 epoch ({max_steps:,} steps)...\n")
results = trainer.train(max_steps=max_steps)

# Save model
save_path = 'models/k1_dataset1.pt'
os.makedirs('models', exist_ok=True)

torch.save({
    'model': trainer.model.state_dict(),  # Save entire model
    'config': config,
    'results': results,
    'vocab_size': trainer.vocab_size,
    # Save vocabulary for continual learning
    'vocab': data_loader.vocab,
    'word_to_idx': data_loader.word_to_idx,
    'idx_to_word': data_loader.idx_to_word
}, save_path)

print(f"\n✓ Model saved to: {save_path}")
print(f"✓ Final perplexity: {results.get('perplexity', 'N/A')}")
print(f"✓ Training time: {results['time']:.1f}s")
