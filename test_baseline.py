#!/usr/bin/env python3
"""
Test Baseline: Traditional Backpropagation Training

This script tests the baseline model independently.
Uses standard gradient descent to update ALL parameters every step.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

import torch
import numpy as np
import json

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import baseline trainer and data loader
from models.baseline_trainer import BaselineTrainer
from data.loader import DataLoader


def load_config():
    """Load configuration."""
    return {
        'model': {
            'vocab_size': 10000,
            'embed_dim': 128,
            'hidden_dim': 256,
            'max_seq_len': 64
        },
        'learning': {
            'learning_rate': 0.001,
            'log_interval': 5000,
            'validation_interval': 5000
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 1
        }
    }


def main():
    print("\n" + "="*70)
    print("BASELINE TEST: Traditional Backpropagation")
    print("="*70)
    
    # Load data
    print("\nLoading WikiText-2 dataset...")
    data_loader = DataLoader(
        dataset_name='wikitext',
        vocab_size=10000,
        seq_length=64
    )
    
    print(f"  Train samples: {len(data_loader.train_data):,}")
    print(f"  Val samples: {len(data_loader.val_data):,}")
    print(f"  Vocabulary size: {data_loader.get_vocab_size():,}")
    
    # Load config
    config = load_config()
    
    # Calculate steps for 1 epoch
    num_epochs = 1
    steps_per_epoch = len(data_loader.train_data)
    max_steps = num_epochs * steps_per_epoch
    print(f"\nTraining for {num_epochs} epoch(s) = {max_steps:,} steps")
    
    # Create baseline trainer
    trainer = BaselineTrainer(config, data_loader=data_loader)
    
    # Train
    results = trainer.train(max_steps=max_steps)
    
    # Print results
    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70)
    print(f"Total steps: {results['total_steps']:,}")
    print(f"Total parameter updates: {results['total_params_updated']:,}")
    print(f"Update percentage: {results['update_percentage']:.1f}%")
    print(f"Training time: {results['time']:.1f}s")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
