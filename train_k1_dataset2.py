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

# Load previous checkpoint to get vocabulary
load_path = 'models/k1_dataset1.pt'
if not os.path.exists(load_path):
    print(f"ERROR: Must run train_k1_dataset1.py first!")
    print(f"No checkpoint found at {load_path}")
    sys.exit(1)

print(f"Loading vocabulary from {load_path}...")
checkpoint = torch.load(load_path, map_location=device)
saved_vocab_size = checkpoint['vocab_size']
print(f"✓ Loaded vocab size: {saved_vocab_size:,}")

# Create a dummy loader to hold the vocab
class VocabHolder:
    pass
shared_vocab = VocabHolder()
shared_vocab.vocab = checkpoint['vocab']
shared_vocab.word_to_idx = checkpoint['word_to_idx']
shared_vocab.idx_to_word = checkpoint['idx_to_word']

# Load Dataset 2: Python Code with SHARED vocabulary
print("\n" + "="*70)
print("DATASET 2: Python Code (Domain Shift from WikiText)")
print("="*70)
print("Loading Python code dataset WITH SHARED VOCABULARY...")

data_loader = DataLoader(
    dataset_name='code_python', 
    vocab_size=saved_vocab_size, 
    seq_length=64,
    shared_vocab=shared_vocab
)

# Get number of samples
if isinstance(data_loader.train_data, tuple):
    num_samples = data_loader.train_data[0].shape[0]
else:
    num_samples = len(data_loader.train_data)

print(f"Train samples: {num_samples:,}")
print(f"Vocab size: {data_loader.get_vocab_size():,} (shared from Dataset 1)\n")

# Create trainer with same vocab size
trainer = HybridK1Trainer(config, data_loader=data_loader)

# Load previous model weights (continual learning!)
print(f"✓ Loading pre-trained model from {load_path}")
trainer.model.load_state_dict(checkpoint['model'])
print("✓ Loaded! Continuing training on CODE domain...\n")

# Train on Python code
max_steps = min(10000, num_samples)
print(f"Training for {max_steps:,} steps...\n")

results = trainer.train(max_steps=max_steps)

# Save updated model with vocab
save_path = 'models/k1_dataset2.pt'
torch.save({
    'model': trainer.model.state_dict(),
    'config': config,
    'results': results,
    'vocab_size': trainer.vocab_size,
    'vocab': data_loader.vocab,
    'word_to_idx': data_loader.word_to_idx,
    'idx_to_word': data_loader.idx_to_word
}, save_path)

print(f"\n✓ Model saved to: {save_path}")
print(f"✓ Training time: {results['time']:.1f}s")
print("\nNOTE: Model learned to handle Python CODE after WikiText")
print("Run generate_k1.py with WikiText prompts to test if it forgot!")
