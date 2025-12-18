#!/usr/bin/env python3
"""
K-1 Self-Learning System vs Baseline GPT Comparison

This script trains both the K-1 Self-Learning model and a baseline GPT model,
then compares their performance across multiple metrics.

Designed to run in Google Colab or any Python environment with numpy.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Import our models
from models.k1_complete import create_k1_complete_model
from models.baseline_gpt_pytorch import BaselineGPTPyTorch

# =============================================================================
# Data Loading
# =============================================================================

def load_wikitext() -> str:
    """Download and load WikiText-2 dataset."""
    print("Loading WikiText-2 dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        # Join train, validation, and test sets
        train_text = "\n".join(dataset['train']['text'])
        val_text = "\n".join(dataset['validation']['text'])
        test_text = "\n".join(dataset['test']['text'])
        
        print(f"Loaded WikiText-2: {len(train_text):,} train chars, {len(val_text):,} val chars")
        return train_text, val_text, test_text
    except ImportError:
        print("Error: 'datasets' library not found. Please run: pip install datasets")
        print("Falling back to Tiny Shakespeare...")
        return download_tiny_shakespeare(), "", ""
    except Exception as e:
        print(f"Error loading WikiText: {e}")
        return download_tiny_shakespeare(), "", ""

def download_tiny_shakespeare() -> str:
    """Download or load Tiny Shakespeare dataset (Fallback)."""
    import urllib.request
    import os

    cache_file = "data/tiny_shakespeare.txt"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        print("Loading cached Tiny Shakespeare...")
        with open(cache_file, 'r') as f:
            return f.read()

    print("Downloading Tiny Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode('utf-8')
        with open(cache_file, 'w') as f:
            f.write(text)
        print(f"Downloaded {len(text):,} characters")
        return text
    except Exception as e:
        print(f"Download failed: {e}")
        return ""



def prepare_data(text_data, config: Dict) -> Tuple[List, List, List, Dict, Dict]:
    """Prepare training, validation, and test data."""
    
    if isinstance(text_data, tuple):
        train_text, val_text, test_text = text_data
        # If fallback matched only one return
        if not val_text:
             # Manual split
             n = len(train_text)
             train_split = int(n * 0.9)
             val_split = int(n * 0.95)
             test_text = train_text[val_split:]
             val_text = train_text[train_split:val_split]
             train_text = train_text[:train_split]
    else:
        # Should not happen with new loader, but safe fallback
        n = len(text_data)
        train_split = int(n * 0.9)
        val_split = int(n * 0.95)
        test_text = text_data[val_split:]
        val_text = text_data[train_split:val_split]
        train_text = text_data[:train_split]

    # Build vocabulary from ALL data
    full_text = train_text + val_text + test_text
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} characters")

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode
    train_encoded = np.array([char_to_idx[ch] for ch in train_text])
    val_encoded = np.array([char_to_idx[ch] for ch in val_text])
    test_encoded = np.array([char_to_idx[ch] for ch in test_text])

    # Create sequences
    seq_len = config['model']['max_seq_len']

    def create_sequences(data):
        sequences = []
        for i in range(0, len(data) - seq_len - 1, seq_len): # Non-overlapping for speed? or seq_len // 2
            x = data[i:i + seq_len]
            y = data[i + 1:i + seq_len + 1]
            if len(x) == seq_len and len(y) == seq_len:
                sequences.append((x, y))
        return sequences

    train_data = create_sequences(train_encoded)
    val_data = create_sequences(val_encoded)
    test_data = create_sequences(test_encoded)

    print(f"Train sequences: {len(train_data)}")
    print(f"Val sequences: {len(val_data)}")
    print(f"Test sequences: {len(test_data)}")
    
    # Convert to PyTorch DataLoaders
    batch_size = 64
    
    def create_dataloader(seq_data, shuffle=True):
        if not seq_data:
            return None
        # Convert to numpy array first to avoid slow tensor creation warning
        x_data = torch.tensor(np.array([s[0] for s in seq_data]), dtype=torch.long)
        y_data = torch.tensor(np.array([s[1] for s in seq_data]), dtype=torch.long)
        dataset = TensorDataset(x_data, y_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0) # num_workers=0 for Colab safety

    train_loader = create_dataloader(train_data, shuffle=True)
    val_loader = create_dataloader(val_data, shuffle=False)
    test_loader = create_dataloader(test_data, shuffle=False)

    return train_loader, val_loader, test_loader, char_to_idx, idx_to_char


# =============================================================================
# Training Functions
# =============================================================================

from models.k1_complete import K1CompleteSystem

def train_k1_model(model: K1CompleteSystem, train_loader: DataLoader, val_loader: DataLoader,
                   config: Dict, verbose: bool = True) -> Dict:
    """Train the K-1 Complete Self-Learning model with trust-based updates."""
    print("\n" + "=" * 60)
    print("Training K-1 Complete Model (Trust-Based Sparse Updates)")
    print("=" * 60)

    max_steps = config['training']['max_steps']
    log_every = config['training']['log_every']
    eval_every = config['training']['eval_every']

    history = {
        'train_loss': [],
        'val_loss': [],
        'improvement': [],
        'num_updated_agents': [],
        'active_agents': [],
        'avg_trust': [],
        'phase': []
    }

    start_time = time.time()
    best_val_loss = float('inf')
    
    step = 0
    epoch = 0
    
    while step < max_steps:
        epoch += 1
        for x_batch, y_batch in train_loader:
            step += 1
            if step > max_steps:
                break
            
            # K-1 trust-based training step
            metrics = model.train_step(x_batch, y_batch)
            
            # Record metrics
            history['train_loss'].append(metrics['loss'])
            history['improvement'].append(metrics['improvement'])
            history['num_updated_agents'].append(metrics['num_updated_agents'])
            history['active_agents'].append(metrics['total_active_agents'])
            history['avg_trust'].append(metrics['avg_trust'])
            history['phase'].append(metrics['phase'])

            # Logging
            if step % log_every == 0 and verbose:
                stats = model.get_stats()
                elapsed = time.time() - start_time
                print(f"Step {step:5d} (Ep {epoch}) | Loss: {metrics['loss']:.4f} | "
                      f"Updated: {metrics['num_updated_agents']}/{metrics['total_active_agents']} agents | "
                      f"Trust: {metrics['avg_trust']:.3f} | "
                      f"Phase: {metrics['phase']} | "
                      f"Time: {elapsed:.1f}s")

            # Evaluation
            if step % eval_every == 0:
                val_loss = evaluate_k1_model(model, val_loader)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if verbose:
                    print(f"  Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print("\nK-1 Training completed in {total_time:.1f}s")
    
    # Print trust system summary
    final_stats = model.get_stats()
    print("\nFinal Trust System Stats:")
    print(f"  High-trust agents: {final_stats['high_trust_agents']}")
    print(f"  Avg trust: {final_stats['avg_trust']:.3f}")

    return history


def evaluate_k1_model(model, dataloader: DataLoader, max_batches: int = 20) -> float:
    """Evaluate K-1 complete model on data."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            
            x = x.to(model.device)
            y = y.to(model.device)
            
            logits = model.forward(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            total_loss += loss.item()
            count += 1
    
    model.train()
    return total_loss / count if count > 0 else 0.0


def train_baseline_model(model: BaselineGPTPyTorch, train_loader: DataLoader, val_loader: DataLoader,
                          config: Dict, verbose: bool = True) -> Dict:
    """Train the baseline GPT model (PyTorch GPU version)."""
    print("\n" + "=" * 60)
    print("Training Baseline GPT Model (PyTorch)")
    print("=" * 60)

    max_steps = config['training']['max_steps']
    log_every = config['training']['log_every']
    eval_every = config['training']['eval_every']

    history = {
        'train_loss': [],
        'val_loss': []
    }

    start_time = time.time()
    best_val_loss = float('inf')
    
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        for x_batch, y_batch in train_loader:
            step += 1
            if step > max_steps:
                break
            
            # Move to GPU and train (PyTorch handles batches properly)
            x_b = x_batch.to(model.device)
            y_b = y_batch.to(model.device)
            
            loss = model.train_step(x_b, y_b)
            history['train_loss'].append(loss)
            
            # Logging
            if step % log_every == 0 and verbose:
                stats = model.get_stats()
                elapsed = time.time() - start_time
                print(f"Step {step:5d} (Ep {epoch}) | Loss: {loss:.4f} | "
                      f"Params: {stats['total_parameters']:,} | "
                      f"Phase: {stats.get('device', 'GPU')} | "
                      f"Time: {elapsed:.1f}s")
            
            # Evaluation
            if step % eval_every == 0:
                val_loss = evaluate_baseline_model(model, val_loader)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if verbose:
                    print(f"  Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nBaseline Training completed in {total_time:.1f}s")

    return history


def evaluate_model(model: K1GPUModel, dataloader: DataLoader, max_batches: int = 20) -> float:
    """Evaluate K-1 model on data."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            
            x = x.to(model.device)
            y = y.to(model.device)
            
            logits = model.forward(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            total_loss += loss.item()
            count += 1
            
    return total_loss / count if count > 0 else 0.0


def evaluate_baseline_model(model: BaselineGPTPyTorch, dataloader: DataLoader, max_batches: int = 20) -> float:
    """Evaluate baseline model on data (PyTorch GPU version)."""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            
            x = x.to(model.device)
            y = y.to(model.device)
            
            logits = model.forward(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model.vocab_size), 
                y.view(-1)
            )
            total_loss += loss.item()
            count += 1
    
    model.train()
    return total_loss / count if count > 0 else 0.0


# =============================================================================
# Comparison and Analysis
# =============================================================================

@dataclass
class ComparisonResults:
    """Results from model comparison."""
    k1_final_loss: float
    baseline_final_loss: float
    k1_best_loss: float
    baseline_best_loss: float
    k1_perplexity: float
    baseline_perplexity: float
    k1_params: int
    baseline_params: int
    k1_training_time: float
    baseline_training_time: float
    k1_sample: str
    baseline_sample: str


from models.k1_complete import K1CompleteSystem

def compare_models(k1_model: K1CompleteSystem, baseline_model: BaselineGPTPyTorch,
                   k1_history: Dict, baseline_history: Dict,
                   test_data: List, idx_to_char: Dict,
                   config: Dict) -> ComparisonResults:
    """Compare both models comprehensively."""
    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)

    # Final losses
    k1_final = np.mean(k1_history['train_loss'][-100:])
    baseline_final = np.mean(baseline_history['train_loss'][-100:])

    # Best validation losses
    k1_best = min(k1_history['val_loss']) if k1_history['val_loss'] else k1_final
    baseline_best = min(baseline_history['val_loss']) if baseline_history['val_loss'] else baseline_final

    # Perplexity
    k1_perplexity = np.exp(k1_best)
    baseline_perplexity = np.exp(baseline_best)

    # Parameters
    k1_stats = k1_model.get_stats()
    baseline_stats = baseline_model.get_stats()
    k1_params = k1_stats['total_parameters']
    baseline_params = baseline_stats['total_parameters']

    # Generate samples
    prompt = np.array([0] * 10)  # Start with some tokens
    k1_generated = k1_model.generate(prompt, max_new_tokens=50)
    baseline_generated = baseline_model.generate(prompt, max_new_tokens=50)

    k1_sample = ''.join([idx_to_char.get(i, '?') for i in k1_generated])
    baseline_sample = ''.join([idx_to_char.get(i, '?') for i in baseline_generated])

    # Print comparison
    print("\n1. LOSS COMPARISON")
    print("-" * 40)
    print(f"  K-1 Final Loss:      {k1_final:.4f}")
    print(f"  Baseline Final Loss: {baseline_final:.4f}")
    print(f"  Winner: {'K-1' if k1_final < baseline_final else 'Baseline'}")

    print("\n2. PERPLEXITY COMPARISON")
    print("-" * 40)
    print(f"  K-1 Perplexity:      {k1_perplexity:.2f}")
    print(f"  Baseline Perplexity: {baseline_perplexity:.2f}")
    print(f"  Winner: {'K-1' if k1_perplexity < baseline_perplexity else 'Baseline'}")

    print("\n3. PARAMETER EFFICIENCY")
    print("-" * 40)
    print(f"  K-1 Parameters:      {k1_params:,}")
    print(f"  Baseline Parameters: {baseline_params:,}")
    k1_efficiency = k1_final / k1_params * 1e6
    baseline_efficiency = baseline_final / baseline_params * 1e6
    print(f"  K-1 Loss/Million Params:      {k1_efficiency:.4f}")
    print(f"  Baseline Loss/Million Params: {baseline_efficiency:.4f}")
    print(f"  More Efficient: {'K-1' if k1_efficiency < baseline_efficiency else 'Baseline'}")

    print("\n4. K-1 SPECIFIC FEATURES")
    print("-" * 40)
    print(f"  Total Agents: {k1_stats['num_agents']}")
    print(f"  Trust Updates: {k1_stats.get('trust_updates', 'N/A')}")
    print(f"  Structural Changes: {k1_stats.get('structural_changes', 'N/A')}")

    print("\n5. GENERATION SAMPLES")
    print("-" * 40)
    print(f"  K-1 Sample:\n    {k1_sample[:100]}...")
    print(f"\n  Baseline Sample:\n    {baseline_sample[:100]}...")

    print("\n6. OVERALL ASSESSMENT")
    print("-" * 40)
    wins = {'K-1': 0, 'Baseline': 0}
    if k1_final < baseline_final:
        wins['K-1'] += 1
    else:
        wins['Baseline'] += 1
    if k1_perplexity < baseline_perplexity:
        wins['K-1'] += 1
    else:
        wins['Baseline'] += 1
    if k1_efficiency < baseline_efficiency:
        wins['K-1'] += 1
    else:
        wins['Baseline'] += 1

    print(f"  K-1 Wins: {wins['K-1']}/3")
    print(f"  Baseline Wins: {wins['Baseline']}/3")

    if wins['K-1'] > wins['Baseline']:
        print("\n  OVERALL WINNER: K-1 Self-Learning Model")
        print("  The hierarchical trust-based approach shows promise!")
    elif wins['Baseline'] > wins['K-1']:
        print("\n  OVERALL WINNER: Baseline GPT Model")
        print("  Standard transformer architecture remains strong.")
    else:
        print("\n  RESULT: TIE")
        print("  Both approaches have their merits.")

    return ComparisonResults(
        k1_final_loss=k1_final,
        baseline_final_loss=baseline_final,
        k1_best_loss=k1_best,
        baseline_best_loss=baseline_best,
        k1_perplexity=k1_perplexity,
        baseline_perplexity=baseline_perplexity,
        k1_params=k1_params,
        baseline_params=baseline_params,
        k1_training_time=0,  # Would need to track
        baseline_training_time=0,
        k1_sample=k1_sample,
        baseline_sample=baseline_sample
    )


def plot_training_curves(k1_history: Dict, baseline_history: Dict):
    """Plot training curves (works in Colab)."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        ax1 = axes[0]
        # Smooth the curves
        window = 100
        k1_smooth = np.convolve(k1_history['train_loss'],
                                np.ones(window)/window, mode='valid')
        baseline_smooth = np.convolve(baseline_history['train_loss'],
                                      np.ones(window)/window, mode='valid')

        ax1.plot(k1_smooth, label='K-1 Self-Learning', alpha=0.8)
        ax1.plot(baseline_smooth, label='Baseline GPT', alpha=0.8)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation loss
        ax2 = axes[1]
        if k1_history['val_loss'] and baseline_history['val_loss']:
            ax2.plot(k1_history['val_loss'], 'o-', label='K-1 Self-Learning')
            ax2.plot(baseline_history['val_loss'], 's-', label='Baseline GPT')
            ax2.set_xlabel('Evaluation Point')
            ax2.set_ylabel('Validation Loss')
            ax2.set_title('Validation Loss Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_comparison.png', dpi=150)
        plt.show()
        print("\nTraining curves saved to 'training_comparison.png'")

    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
        print("Training curves data is available in the history dictionaries.")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("K-1 Self-Learning System vs Baseline GPT")
    print("Comparative Training and Evaluation")
    print("=" * 60)

    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("\nLoaded configuration from config.json")
    except FileNotFoundError:
        print("\nUsing default configuration")
        config = {
            'model': {
                'vocab_size': 256,
                'embed_dim': 128,
                'hidden_dim': 256,
                'num_heads': 4,
                'ff_dim': 512,
                'max_seq_len': 64
            },
            'training': {
                'learning_rate': 0.0001,
                'max_steps': 80040,  # 30 Epochs * 2668 steps/epoch
                'log_every': 500,
                'eval_every': 2668   # Every epoch
            },
            'k1_system': {
                'hierarchy': {'depth': 3, 'branching_factor': 4},
                'routing': {'top_k': 4, 'exploration_rate': 0.1},
                'trust': {'initial_trust': 0.5}
            },
            'data': {
                'train_split': 0.9,
                'val_split': 0.05,
                'test_split': 0.05
            }
        }

    # Load data
    print("\n" + "-" * 40)
    print("Loading Data (WikiText-2) [Scaled Training: 30 Epochs]")
    print("-" * 40)
    text_data = load_wikitext()
    # Now returns loaders
    train_loader, val_loader, test_loader, char_to_idx, idx_to_char = prepare_data(text_data, config)

    # Update vocab size based on actual data
    actual_vocab_size = len(char_to_idx)
    config['model']['vocab_size'] = actual_vocab_size

    # Initialize models
    print("\n" + "-" * 40)
    print("Initializing Models")
    print("-" * 40)

    # K-1 Complete Model (Trust-based, Sparse Updates)
    print("Initializing K-1 Complete Model with Trust System...")
    
    k1_config = {
        'vocab_size': actual_vocab_size,
        'embed_dim': config['model']['embed_dim'],
        'hidden_dim': config['model']['hidden_dim'],
        'max_seq_len': config['model']['max_seq_len'],
        'learning_rate': config['training']['learning_rate'],
        'top_k': config['k1_system']['credit_assignment']['top_k_agents'],
        'phase_1_duration': 10000,  # Phase 2 activates at 10k iterations
    }
    
    k1_model = create_k1_complete_model(k1_config)
    k1_stats = k1_model.get_stats()
    print(f"K-1 Complete Model: {k1_stats['total_parameters']:,} parameters, {k1_stats['total_agents']} agents")
    print(f"  Trust System: {k1_stats['high_trust_agents']} high-trust agents cached")
    print(f"  Sparse Updates: Top-{k1_config['top_k']} agents per step")

    # Baseline Model - MATCHED to K-1 dimensions for fair comparison
    # Same embed_dim (128) and similar hidden dims to K-1's architecture
    # This ensures both models have similar computational cost per step
    baseline_config = {
        'vocab_size': actual_vocab_size,
        'embed_dim': 128,       # MATCHED to K-1's embed_dim
        'num_layers': 12,       # Moderate depth for comparison
        'num_heads': 4,         # MATCHED to K-1's config
        'ff_dim': 512,          # MATCHED to K-1's ff_dim  
        'max_seq_len': config['model']['max_seq_len'],
        'learning_rate': config['training']['learning_rate'],
        'dropout': 0.1,
        'warmup_steps': 1000
    }
    baseline_model = BaselineGPTPyTorch(baseline_config)
    baseline_stats = baseline_model.get_stats()
    print(f"Baseline Model (PyTorch): {baseline_stats['total_parameters']:,} parameters")

    # Train both models
    print("\n" + "-" * 40)
    print("Starting Training")
    print("-" * 40)

    k1_start = time.time()
    k1_history = train_k1_model(k1_model, train_loader, val_loader, config)
    k1_time = time.time() - k1_start

    baseline_start = time.time()
    baseline_history = train_baseline_model(baseline_model, train_loader, val_loader, config)
    baseline_time = time.time() - baseline_start

    # Compare models
    results = compare_models(
        k1_model, baseline_model,
        k1_history, baseline_history,
        test_loader, idx_to_char, config
    )

    # Add timing
    print(f"\n7. TRAINING TIME")
    print("-" * 40)
    print(f"  K-1 Training Time:      {k1_time:.1f}s")
    print(f"  Baseline Training Time: {baseline_time:.1f}s")

    # Plot curves
    plot_training_curves(k1_history, baseline_history)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nK-1 Self-Learning Model:")
    print(f"  - Uses hierarchical agent structure with trust-based routing")
    print(f"  - Implements sparse updates (only top-K agents per step)")
    print(f"  - Features structural plasticity (pruning, merging, growing)")
    print(f"  - Two-phase learning: fixed then autonomous")

    print(f"\nBaseline GPT Model:")
    print(f"  - Standard transformer architecture")
    print(f"  - Full backpropagation through all parameters")
    print(f"  - Fixed architecture throughout training")

    print(f"\nKey Insight:")
    if results.k1_final_loss < results.baseline_final_loss:
        print("  The K-1 approach shows promise! Trust-based credit assignment")
        print("  combined with sparse updates can compete with full backprop.")
    else:
        print("  Standard backpropagation remains highly effective.")
        print("  The K-1 approach needs more tuning to match transformer performance.")

    return k1_model, baseline_model, k1_history, baseline_history, results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run the experiment
    k1_model, baseline_model, k1_history, baseline_history, results = main()
