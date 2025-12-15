#!/usr/bin/env python3
"""
K-1 Self-Learning System vs Baseline GPT Comparison

This script trains both the K-1 Self-Learning model and a baseline GPT model,
then compares their performance across multiple metrics.

Designed to run in Google Colab or any Python environment with numpy.
"""

import numpy as np
import json
import time
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import our models
from models.k1_model import K1SelfLearningLM
from models.baseline_gpt import BaselineGPT

# =============================================================================
# Data Loading
# =============================================================================

def download_tiny_shakespeare() -> str:
    """Download or load Tiny Shakespeare dataset."""
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
        print("Using synthetic data instead...")
        return generate_synthetic_data()


def generate_synthetic_data(size: int = 100000) -> str:
    """Generate synthetic text data for testing."""
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "To be or not to be, that is the question. ",
        "All that glitters is not gold. ",
        "A journey of a thousand miles begins with a single step. ",
        "Knowledge is power, but wisdom is supreme. ",
    ]
    text = ""
    while len(text) < size:
        text += np.random.choice(patterns)
    return text[:size]


def prepare_data(text: str, config: Dict) -> Tuple[List, List, List, Dict, Dict]:
    """Prepare training, validation, and test data.

    Returns:
        train_data, val_data, test_data, char_to_idx, idx_to_char
    """
    # Build vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} characters")

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode text
    encoded = np.array([char_to_idx[ch] for ch in text])

    # Split data
    n = len(encoded)
    train_split = int(n * config['data']['train_split'])
    val_split = int(n * (config['data']['train_split'] + config['data']['val_split']))

    train_encoded = encoded[:train_split]
    val_encoded = encoded[train_split:val_split]
    test_encoded = encoded[val_split:]

    # Create sequences
    seq_len = config['model']['max_seq_len']

    def create_sequences(data):
        sequences = []
        for i in range(0, len(data) - seq_len - 1, seq_len // 2):
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

    return train_data, val_data, test_data, char_to_idx, idx_to_char


# =============================================================================
# Training Functions
# =============================================================================

def train_k1_model(model: K1SelfLearningLM, train_data: List, val_data: List,
                   config: Dict, verbose: bool = True) -> Dict:
    """Train the K-1 Self-Learning model.

    Returns:
        Training history and metrics
    """
    print("\n" + "=" * 60)
    print("Training K-1 Self-Learning Model")
    print("=" * 60)

    max_steps = config['training']['max_steps']
    log_every = config['training']['log_every']
    eval_every = config['training']['eval_every']

    history = {
        'train_loss': [],
        'val_loss': [],
        'trust_updates': [],
        'structural_changes': [],
        'phase': []
    }

    start_time = time.time()
    best_val_loss = float('inf')

    for step in range(1, max_steps + 1):
        # Sample random training sequence
        idx = np.random.randint(len(train_data))
        x, y = train_data[idx]

        # Training step
        loss = model.train_step(x, y)
        history['train_loss'].append(loss)

        # Get current phase
        phase = model.get_current_phase()
        history['phase'].append(phase)

        # Logging
        if step % log_every == 0 and verbose:
            stats = model.get_stats()
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss:.4f} | "
                  f"Agents: {stats['num_agents']} | "
                  f"Phase: {phase} | "
                  f"Time: {elapsed:.1f}s")

        # Evaluation
        if step % eval_every == 0:
            val_loss = evaluate_model(model, val_data[:100])
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if verbose:
                print(f"  Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nK-1 Training completed in {total_time:.1f}s")

    return history


def train_baseline_model(model: BaselineGPT, train_data: List, val_data: List,
                         config: Dict, verbose: bool = True) -> Dict:
    """Train the baseline GPT model.

    Returns:
        Training history and metrics
    """
    print("\n" + "=" * 60)
    print("Training Baseline GPT Model")
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

    for step in range(1, max_steps + 1):
        # Sample random training sequence
        idx = np.random.randint(len(train_data))
        x, y = train_data[idx]

        # Training step
        loss = model.train_step(x, y)
        history['train_loss'].append(loss)

        # Logging
        if step % log_every == 0 and verbose:
            stats = model.get_stats()
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss:.4f} | "
                  f"Params: {stats['total_parameters']:,} | "
                  f"Time: {elapsed:.1f}s")

        # Evaluation
        if step % eval_every == 0:
            val_loss = evaluate_baseline_model(model, val_data[:100])
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if verbose:
                print(f"  Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nBaseline Training completed in {total_time:.1f}s")

    return history


def evaluate_model(model: K1SelfLearningLM, data: List) -> float:
    """Evaluate K-1 model on data."""
    total_loss = 0.0
    for x, y in data:
        logits = model.forward(x)
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)
        # Cross-entropy
        loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-10))
        total_loss += loss
    return total_loss / len(data)


def evaluate_baseline_model(model: BaselineGPT, data: List) -> float:
    """Evaluate baseline model on data."""
    total_loss = 0.0
    for x, y in data:
        logits, _ = model.forward(x)
        loss, _ = model.compute_loss(logits, y)
        total_loss += loss
    return total_loss / len(data)


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


def compare_models(k1_model: K1SelfLearningLM, baseline_model: BaselineGPT,
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
                'max_steps': 5000,
                'log_every': 100,
                'eval_every': 500
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
    print("Loading Data")
    print("-" * 40)
    text = download_tiny_shakespeare()
    train_data, val_data, test_data, char_to_idx, idx_to_char = prepare_data(text, config)

    # Update vocab size based on actual data
    actual_vocab_size = len(char_to_idx)
    config['model']['vocab_size'] = actual_vocab_size

    # Initialize models
    print("\n" + "-" * 40)
    print("Initializing Models")
    print("-" * 40)

    # K-1 Model
    k1_config = {
        'vocab_size': actual_vocab_size,
        'embed_dim': config['model']['embed_dim'],
        'hidden_dim': config['model']['hidden_dim'],
        'hierarchy_depth': config['k1_system']['hierarchy']['depth'],
        'branching_factor': config['k1_system']['hierarchy']['branching_factor'],
        'top_k_routing': config['k1_system']['routing']['top_k'],
        'learning_rate': config['training']['learning_rate'],
        'exploration_rate': config['k1_system']['routing']['exploration_rate'],
        'initial_trust': config['k1_system']['trust']['initial_trust'],
        'phase1_steps': config['training']['max_steps'] // 2,
        'phase2_steps': config['training']['max_steps'] // 2
    }
    k1_model = K1SelfLearningLM(k1_config)
    k1_stats = k1_model.get_stats()
    print(f"K-1 Model: {k1_stats['total_parameters']:,} parameters, {k1_stats['num_agents']} agents")

    # Baseline Model
    baseline_config = {
        'vocab_size': actual_vocab_size,
        'embed_dim': config['model']['embed_dim'],
        'num_layers': config['k1_system']['hierarchy']['depth'],  # Match depth
        'num_heads': config['model']['num_heads'],
        'ff_dim': config['model']['ff_dim'],
        'max_seq_len': config['model']['max_seq_len'],
        'learning_rate': config['training']['learning_rate']
    }
    baseline_model = BaselineGPT(baseline_config)
    baseline_stats = baseline_model.get_stats()
    print(f"Baseline Model: {baseline_stats['total_parameters']:,} parameters")

    # Train both models
    print("\n" + "-" * 40)
    print("Starting Training")
    print("-" * 40)

    k1_start = time.time()
    k1_history = train_k1_model(k1_model, train_data, val_data, config)
    k1_time = time.time() - k1_start

    baseline_start = time.time()
    baseline_history = train_baseline_model(baseline_model, train_data, val_data, config)
    baseline_time = time.time() - baseline_start

    # Compare models
    results = compare_models(
        k1_model, baseline_model,
        k1_history, baseline_history,
        test_data, idx_to_char, config
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
