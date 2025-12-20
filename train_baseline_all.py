#!/usr/bin/env python3
"""
Train Baseline on all 3 datasets sequentially
Tests catastrophic forgetting in standard backprop

FAIR COMPARISON: Uses SAME architecture as K-1 (ModularSparseTransformer)
Only difference: Dense updates (all params) vs Sparse updates (selected groups)
"""

import sys
import os
import json
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.baseline_trainer import BaselineTrainer
from data.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load same config as K-1 for fair comparison
config_path = os.path.join(project_root, 'k1_system/config/config_phase1.json')
with open(config_path, 'r') as f:
    config = json.load(f)

print("\n" + "="*70)
print("BASELINE: Training on 3 Datasets Sequentially")
print("="*70)
print("Using SAME architecture as K-1 (ModularSparseTransformer)")
print("Difference: Dense updates (ALL params) vs K-1's Sparse updates\n")

# Dataset 1
print("="*70)
print("DATASET 1: WikiText-2")
print("="*70)
data_loader1 = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)

# Get number of samples
if isinstance(data_loader1.train_data, tuple):
    max_steps1 = data_loader1.train_data[0].shape[0]
else:
    max_steps1 = len(data_loader1.train_data)

print(f"Train samples: {max_steps1:,}")
print(f"Vocab size: {data_loader1.get_vocab_size():,}\n")

trainer = BaselineTrainer(config, data_loader=data_loader1)

print(f"Training for 1 epoch ({max_steps1:,} steps)...\n")
results1 = trainer.train(max_steps=max_steps1)

# Save after dataset 1 (with vocab for continual learning)
save_path1 = 'models/baseline_dataset1.pt'
os.makedirs('models', exist_ok=True)
torch.save({
    'model': trainer.model.state_dict(),
    'vocab_size': trainer.vocab_size,
    'results': results1,
    'vocab': data_loader1.vocab,
    'word_to_idx': data_loader1.word_to_idx,
    'idx_to_word': data_loader1.idx_to_word
}, save_path1)
print(f"✓ Saved to {save_path1}\n")

# Dataset 2 (continue training with shared vocab)
print("\n" + "="*70)
print("DATASET 2: Python Code (Continuing...)")
print("="*70)
print("⚠ Standard backprop may forget Dataset 1!\n")

# Use shared vocab
class VocabHolder:
    pass
shared_vocab = VocabHolder()
shared_vocab.vocab = data_loader1.vocab
shared_vocab.word_to_idx = data_loader1.word_to_idx
shared_vocab.idx_to_word = data_loader1.idx_to_word

data_loader2 = DataLoader(
    dataset_name='code_python', 
    vocab_size=10000, 
    seq_length=64,
    shared_vocab=shared_vocab
)
trainer.data_loader = data_loader2

# Get number of samples
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
    'model': trainer.model.state_dict(),
    'vocab_size': trainer.vocab_size,
    'results': results2,
    'vocab': data_loader1.vocab,
    'word_to_idx': data_loader1.word_to_idx,
    'idx_to_word': data_loader1.idx_to_word
}, save_path2)
print(f"✓ Saved to {save_path2}\n")

# Dataset 3
print("\n" + "="*70)
print("DATASET 3: Scientific Text (Final...)")
print("="*70)

data_loader3 = DataLoader(
    dataset_name='scientific', 
    vocab_size=10000, 
    seq_length=64,
    shared_vocab=shared_vocab
)
trainer.data_loader = data_loader3

# Get number of samples
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
    'model': trainer.model.state_dict(),
    'vocab_size': trainer.vocab_size,
    'config': config,
    'results': results3,
    'vocab': data_loader1.vocab,
    'word_to_idx': data_loader1.word_to_idx,
    'idx_to_word': data_loader1.idx_to_word
}, save_path_final)
print(f"✓ Final model saved to {save_path_final}\n")

print("="*70)
print("BASELINE TRAINING COMPLETE")
print("="*70)
print("Compare with K-1 using generate scripts to test forgetting!")
