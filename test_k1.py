#!/usr/bin/env python3
"""
Test K-1: Hybrid Self-Learning System

This script tests the K-1 system independently.
Uses sparse gradient updates with trust-based selection.
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

# Import K-1 trainer and data loader
from k1_system.learning.hybrid_trainer import HybridK1Trainer
from data.loader import DataLoader


def load_config():
    """Load K-1 configuration."""
    config_path = os.path.join(project_root, 'k1_system/config/config_phase1.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    print("\n" + "="*70)
    print("K-1 TEST: Hybrid Self-Learning System")
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
    
    # Create K-1 trainer
    trainer = HybridK1Trainer(config, data_loader=data_loader)
    
    # Train
    results = trainer.train(max_steps=max_steps)
    
    # Print results
    print("\n" + "="*70)
    print("K-1 RESULTS")
    print("="*70)
    print(f"Total steps: {results['total_steps']:,}")
    print(f"Total parameter updates: {results['total_params_updated']:,}")
    print(f"Avg params per step: {results['avg_params_per_step']:,}")
    print(f"Update percentage: {results['update_percentage']:.1f}%")
    print(f"Phase 2 adjustments: {results.get('phase_2_adjustments', 0)}")
    print(f"Training time: {results['time']:.1f}s")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
