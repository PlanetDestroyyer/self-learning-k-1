#!/usr/bin/env python3
"""
Comparison: Baseline vs Hybrid K-1 System

This script compares traditional backpropagation against the K-1 hybrid approach.
Both trainers use the same data loading and cross-entropy loss for fair comparison.

BASELINE:
- Updates ALL parameters every step (100%)
- Standard gradient descent

HYBRID K-1:
- Computes gradients for ALL agents
- Updates only TOP-K selected by: GRADIENT + TRUST + DIVERSITY
- Phase 1: Gradient-based exploration
- Phase 2: Autonomous adaptation
"""

import torch
import numpy as np
import json
import sys
import os

# Add project root to path
if '__file__' in globals():
    project_root = os.path.dirname(os.path.abspath(__file__))
else:
    project_root = os.getcwd()

for path in [project_root, os.path.join(project_root, '..'), '/content/self-learning-k-1']:
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Import trainers and data loading
try:
    from k1_system.learning import HybridK1Trainer
    from models.baseline_trainer import BaselineTrainer
    from data.loader import DataLoader
    print("✓ All imports successful")
except ImportError as e:
    print(f"ERROR: Failed to import: {e}")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)


def load_config(use_k1: bool = True) -> dict:
    """Load configuration."""
    if use_k1:
        with open('k1_system/config/config_phase1.json', 'r') as f:
            return json.load(f)
    else:
        return {
            'model': {
                'vocab_size': 10000,
                'embed_dim': 128,
                'hidden_dim': 256
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 1
            }
        }


def load_wikitext_data(vocab_size: int = 10000, seq_length: int = 64):
    """Load WikiText-2 dataset."""
    print("\n" + "="*70)
    print("Loading WikiText-2 Dataset")
    print("="*70)
    
    try:
        data_loader = DataLoader(
            dataset_name='wikitext',
            vocab_size=vocab_size,
            seq_length=seq_length,
            use_real_data=True
        )
        
        train_size = len(data_loader.train_data)
        val_size = len(data_loader.val_data) if data_loader.val_data is not None else 0
        test_size = len(data_loader.test_data) if data_loader.test_data is not None else 0
        
        print(f"Successfully loaded WikiText-2:")
        print(f"  Train samples: {train_size:,}")
        print(f"  Val samples: {val_size:,}")
        print(f"  Test samples: {test_size:,}")
        print(f"  Vocabulary size: {data_loader.get_vocab_size():,}")
        print("="*70 + "\n")
        
        return data_loader
        
    except Exception as e:
        print(f"Failed to load WikiText-2: {e}")
        print("="*70 + "\n")
        return None


def generate_synthetic_data(n_samples: int = 500, dim: int = 128) -> np.ndarray:
    """Generate synthetic data for testing."""
    return np.random.randn(n_samples, dim).astype(np.float32)


def compare_approaches(num_epochs: int = 1, use_wikitext: bool = True):
    """Run comparison between baseline and hybrid K-1."""
    print("\n" + "="*70)
    print("COMPARISON: Baseline vs Hybrid K-1")
    print("="*70)
    print(f"Training: {num_epochs} epoch(s)")
    print("="*70)
    
    # Load data
    data_loader = None
    if use_wikitext:
        data_loader = load_wikitext_data(vocab_size=10000, seq_length=64)
    
    if data_loader is None:
        data = generate_synthetic_data(n_samples=500)
        print(f"Using synthetic data: {data.shape[0]} samples\n")
        num_samples = 500
    else:
        data = None
        num_samples = len(data_loader.train_data)
    
    # Calculate training steps
    max_steps = num_epochs * num_samples
    print(f"Dataset size: {num_samples:,} samples")
    print(f"Total training steps: {max_steps:,}")
    
    # =========================================================================
    # BASELINE
    # =========================================================================
    baseline_config = load_config(use_k1=False)
    baseline_trainer = BaselineTrainer(baseline_config, data_loader=data_loader)
    baseline_results = baseline_trainer.train(data, max_steps=max_steps)
    
    # =========================================================================
    # HYBRID K-1
    # =========================================================================
    k1_config = load_config(use_k1=True)
    # Phase 2 activates after 5000 steps or half, whichever is smaller
    k1_config['system']['phase_1_duration'] = min(5000, max_steps // 2)
    k1_config['training']['max_steps'] = max_steps
    
    if data_loader is not None:
        k1_config['model']['vocab_size'] = data_loader.get_vocab_size()
    
    k1_trainer = HybridK1Trainer(k1_config, data_loader=data_loader)
    k1_results = k1_trainer.train(data, max_steps=max_steps)
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print_comparison(baseline_results, k1_results)


def print_comparison(baseline: dict, k1: dict):
    """Print side-by-side comparison."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<40} {'Baseline':>15} {'Hybrid K-1':>15}")
    print("-"*70)
    
    print(f"{'Total steps':<40} {baseline['total_steps']:>15,} {k1['total_steps']:>15,}")
    print(f"{'Total parameter updates':<40} {baseline['total_params_updated']:>15,} {k1['total_params_updated']:>15,}")
    print(f"{'Avg params updated per step':<40} {baseline['avg_params_per_step']:>15,.0f} {k1['avg_params_per_step']:>15,.0f}")
    print(f"{'Update percentage':<40} {100.0:>14.1f}% {k1.get('update_percentage', 0):>14.1f}%")
    
    reduction = (1 - k1['total_params_updated'] / baseline['total_params_updated']) * 100
    print(f"{'Parameter update reduction':<40} {'N/A':>15} {reduction:>14.1f}%")
    
    if 'phase_2_adjustments' in k1:
        print(f"{'Phase 2 adjustments':<40} {'N/A':>15} {k1['phase_2_adjustments']:>15}")
    
    print(f"{'Training time':<40} {baseline['time']:>14.1f}s {k1['time']:>14.1f}s")
    
    print("="*70)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print(f"""
BASELINE (Traditional Backprop):
  - Updates 100% of parameters every step
  - No selectivity or interpretability

HYBRID K-1 (Gradient + Trust + Diversity):
  - Updates only {k1.get('update_percentage', 5):.1f}% of parameters per step
  - Uses REAL gradients (not heuristics)
  - Trust prevents "rich get richer" problem
  - Autonomous structural adaptation in Phase 2
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run comparison with WikiText-2
    # Use 1 epoch for quick testing, increase for longer training
    compare_approaches(num_epochs=1, use_wikitext=True)
