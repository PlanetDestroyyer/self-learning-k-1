#!/usr/bin/env python3
"""
Train K-1 Hierarchical System

This trains the TRUE K-1 system with:
- Hierarchical tree of nodes (Manager → Agent → Sub-agent)
- Path-based gradient updates (only update responsible paths)
- Continual learning capability
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader
from k1_system.training import HierarchicalK1Trainer

import torch


def main():
    print("=" * 70)
    print("K-1 HIERARCHICAL SYSTEM - True Path-Based Learning")
    print("=" * 70)
    
    # Load config
    config_path = Path(__file__).parent / 'k1_system' / 'config' / 'config_phase1.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Update config for hierarchical system
    config['model']['tree_depth'] = 3        # Root + 2 levels
    config['model']['branching_factor'] = 3  # 3 children per node
    config['learning']['top_k'] = 5          # Update top 5 nodes per step
    
    # Load data
    print("\nLoading WikiText-2 dataset...")
    data_loader = DataLoader(
        dataset_name='wikitext',
        vocab_size=config['model'].get('vocab_size', 10000),
        seq_length=config['model'].get('max_seq_len', 64)
    )
    
    print(f"Vocabulary size: {data_loader.get_vocab_size()}")
    
    # Create trainer
    print("\nInitializing K-1 Hierarchical System...")
    trainer = HierarchicalK1Trainer(config, data_loader)
    
    # Train
    max_steps = config['training'].get('max_steps', 10000)
    print(f"\nTraining for {max_steps} steps...")
    
    results = trainer.train(max_steps=max_steps)
    
    # Save checkpoint
    checkpoint_path = Path(__file__).parent / 'checkpoints' / 'k1_hierarchical.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'vocab': {
            'word_to_idx': data_loader.word_to_idx,
            'idx_to_word': data_loader.idx_to_word,
            'vocab': data_loader.vocab,
        },
        'config': config,
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print(f"Final loss: {results['loss']:.4f}")
    print(f"Training time: {results['time']:.1f}s")


if __name__ == '__main__':
    main()
