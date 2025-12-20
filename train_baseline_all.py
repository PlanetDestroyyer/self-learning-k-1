#!/usr/bin/env python3
"""
Train Baseline on all 3 datasets sequentially
Tests catastrophic forgetting in standard backprop
"""

import sys
import os
import json
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.baseline_trainer import BaselineTrainer
from data.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Config
config = {
    'model': {'vocab_size': 10000, 'embed_dim': 128, 'hidden_dim': 256},
    'training': {'learning_rate': 0.001},
    'learning': {'batch_size': 256, 'log_interval': 5000, 'validation_interval': 5000}
}

print("="*70)
print("BASELINE: Training on 3 Datasets Sequentially")
print("="*70)
print("This tests catastrophic forgetting in standard backpropagation\n")

# Dataset 1
print("\n" + "="*70)
print("DATASET 1: WikiText-2")
print("="*70)
data_loader1 = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)
trainer = BaselineTrainer(config, data_loader=data_loader1)

# Get number of samples (handle both list and GPU tensor formats)
if isinstance(data_loader1.train_data, tuple):
    max_steps1 = data_loader1.train_data[0].shape[0]
else:
    max_steps1 = len(data_loader1.train_data)
print(f"Training for {max_steps1:,} steps...\n")
results1 = trainer.train(max_steps=max_steps1)

# Save after dataset 1
save_path1 = 'models/baseline_dataset1.pt'
os.makedirs('models', exist_ok=True)
torch.save({
    'embedding': trainer.embedding.state_dict(),
    'layers': trainer.layers.state_dict(),
    'output_proj': trainer.output_proj.state_dict(),
    'vocab_size': trainer.vocab_size,
    'results': results1
}, save_path1)
print(f"✓ Saved to {save_path1}\n")

# Dataset 2 (continue training - will it forget Dataset 1?)
print("\n" + "="*70)
print("DATASET 2: Continuing training...")
print("="*70)
print("⚠ Standard backprop may forget Dataset 1!\n")

data_loader2 = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)
trainer.data_loader = data_loader2

# Get number of samples (handle both list and GPU tensor formats)
if isinstance(data_loader2.train_data, tuple):
    num_samples2 = data_loader2.train_data[0].shape[0]
else:
    num_samples2 = len(data_loader2.train_data)
max_steps2 = min(10000, num_samples2)
print(f"Training for {max_steps2:,} steps...\n")
results2 = trainer.train(max_steps=max_steps2)

# Save after dataset 2
save_path2 = 'models/baseline_dataset2.pt'
torch.save({
    'embedding': trainer.embedding.state_dict(),
    'layers': trainer.layers.state_dict(),
    'output_proj': trainer.output_proj.state_dict(),
    'vocab_size': trainer.vocab_size,
    'results': results2
}, save_path2)
print(f"✓ Saved to {save_path2}\n")

# Dataset 3
print("\n" + "="*70)
print("DATASET 3: Final training...")
print("="*70)

data_loader3 = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)
trainer.data_loader = data_loader3

# Get number of samples (handle both list and GPU tensor formats)
if isinstance(data_loader3.train_data, tuple):
    num_samples3 = data_loader3.train_data[0].shape[0]
else:
    num_samples3 = len(data_loader3.train_data)
max_steps3 = min(10000, num_samples3)
print(f"Training for {max_steps3:,} steps...\n")
results3 = trainer.train(max_steps=max_steps3)

# Save final
save_path_final = 'models/baseline_final.pt'
torch.save({
    'embedding': trainer.embedding.state_dict(),
    'layers': trainer.layers.state_dict(),
    'output_proj': trainer.output_proj.state_dict(),
    'vocab_size': trainer.vocab_size,
    'results': results3
}, save_path_final)
print(f"✓ Final model saved to {save_path_final}\n")

print("="*70)
print("BASELINE TRAINING COMPLETE")
print("="*70)
print("Compare with K-1 using generate scripts to test forgetting!")
