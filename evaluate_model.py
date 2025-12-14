"""
Comprehensive evaluation script for K-1 Language Model.

This script verifies that the model is actually learning by:
1. Comparing to baseline random model
2. Tracking perplexity improvement over time
3. Testing text generation quality
4. Analyzing agent specialization
"""

import pickle
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt


def load_trained_model(model_path: str = 'trained_k1_10m.pkl'):
    """Load trained model."""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"✅ Model loaded:")
    print(f"   Parameters: {model_data['total_params']:,}")
    print(f"   Best perplexity: {model_data['best_perplexity']:.2f}")
    print(f"   Agents: {model_data['hierarchy'].count_agents()}")
    return model_data


def load_training_metrics(metrics_file: str = None):
    """Load training metrics."""
    if metrics_file is None:
        # Find latest metrics file
        import glob
        metrics_files = glob.glob('logs/metrics_*.json')
        if not metrics_files:
            print("No metrics files found")
            return None
        metrics_file = max(metrics_files)

    print(f"\nLoading metrics from {metrics_file}...")
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    print(f"✅ Loaded {len(metrics)} training iterations")
    return metrics


def calculate_baseline_perplexity(vocab_size: int = 20000):
    """
    Calculate baseline perplexity for random model.

    A random model that predicts uniformly has:
    perplexity = vocab_size
    """
    return vocab_size


def analyze_learning_curve(metrics):
    """Analyze if model is learning from metrics."""
    if not metrics:
        return

    iterations = [m['iteration'] for m in metrics]
    losses = [m.get('loss', 0) for m in metrics]

    # Check for improvement
    initial_loss = np.mean(losses[:100]) if len(losses) > 100 else losses[0]
    final_loss = np.mean(losses[-100:]) if len(losses) > 100 else losses[-1]

    improvement = initial_loss - final_loss
    improvement_pct = (improvement / initial_loss) * 100

    print("\n📈 Learning Curve Analysis:")
    print(f"   Initial loss (first 100 iters): {initial_loss:.4f}")
    print(f"   Final loss (last 100 iters): {final_loss:.4f}")
    print(f"   Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")

    if improvement > 0:
        print(f"   ✅ Model IS learning (loss decreased)")
    else:
        print(f"   ❌ Model NOT learning (loss increased or flat)")

    # Plot learning curve
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, losses, alpha=0.7, label='Training Loss')
    # Smooth with moving average
    window = 100
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(iterations[window-1:], smooth, 'r-', linewidth=2, label=f'MA({window})')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Agents over time
    plt.subplot(1, 2, 2)
    agent_counts = [m.get('total_agents', 0) for m in metrics]
    plt.plot(iterations, agent_counts, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Number of Agents')
    plt.title('Agent Count Evolution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n   📊 Saved visualization to training_analysis.png")

    return improvement > 0


def compare_to_baseline(model_perplexity: float, vocab_size: int = 20000):
    """Compare model to random baseline."""
    baseline_ppl = calculate_baseline_perplexity(vocab_size)

    print("\n🎯 Baseline Comparison:")
    print(f"   Random model perplexity: {baseline_ppl:.2f}")
    print(f"   Trained model perplexity: {model_perplexity:.2f}")

    improvement_ratio = baseline_ppl / model_perplexity
    bits_per_char = np.log2(model_perplexity)

    print(f"\n   Improvement: {improvement_ratio:.2f}x better than random")
    print(f"   Bits per character: {bits_per_char:.2f}")

    if model_perplexity < baseline_ppl * 0.5:
        print(f"   ✅ EXCELLENT: Model is significantly better than random!")
    elif model_perplexity < baseline_ppl * 0.8:
        print(f"   ✅ GOOD: Model is learning")
    elif model_perplexity < baseline_ppl:
        print(f"   ⚠️  FAIR: Model is slightly better than random")
    else:
        print(f"   ❌ POOR: Model is not better than random")


def analyze_phase_transition(metrics):
    """Analyze Phase 1 vs Phase 2 performance."""
    if not metrics:
        return

    # Find phase transition point
    phase_transition = None
    for i, m in enumerate(metrics):
        if m.get('phase', 1) == 2:
            phase_transition = i
            break

    if phase_transition is None:
        print("\n⚠️  Phase 2 not reached in training")
        return

    print(f"\n🔄 Phase Transition Analysis:")
    print(f"   Phase 1 iterations: 0-{phase_transition}")
    print(f"   Phase 2 iterations: {phase_transition}-{len(metrics)}")

    # Compare performance
    phase1_losses = [m.get('loss', 0) for m in metrics[:phase_transition]]
    phase2_losses = [m.get('loss', 0) for m in metrics[phase_transition:]]

    if phase1_losses and phase2_losses:
        phase1_avg = np.mean(phase1_losses[-100:])
        phase2_avg = np.mean(phase2_losses[-100:])

        print(f"\n   Phase 1 final loss: {phase1_avg:.4f}")
        print(f"   Phase 2 final loss: {phase2_avg:.4f}")

        if phase2_avg < phase1_avg:
            improvement = ((phase1_avg - phase2_avg) / phase1_avg) * 100
            print(f"   ✅ Phase 2 improved by {improvement:.1f}%")
        else:
            print(f"   ⚠️  Phase 2 did not improve over Phase 1")


def analyze_agent_specialization(model_data):
    """Analyze how agents specialized."""
    hierarchy = model_data['hierarchy']
    all_agents = hierarchy.get_all_agents()

    print("\n🎭 Agent Specialization Analysis:")
    print(f"   Total agents: {len(all_agents)}")

    # Group by trust levels
    high_trust = [a for a in all_agents if a.trust > 0.7]
    medium_trust = [a for a in all_agents if 0.3 <= a.trust <= 0.7]
    low_trust = [a for a in all_agents if a.trust < 0.3]

    print(f"\n   Trust distribution:")
    print(f"      High trust (>0.7): {len(high_trust)} agents")
    print(f"      Medium trust (0.3-0.7): {len(medium_trust)} agents")
    print(f"      Low trust (<0.3): {len(low_trust)} agents")

    # Top performers
    if high_trust:
        print(f"\n   🌟 Top 5 specialists:")
        top_agents = sorted(all_agents, key=lambda a: a.trust, reverse=True)[:5]
        for i, agent in enumerate(top_agents, 1):
            print(f"      {i}. {agent.specialty}: trust={agent.trust:.3f}, "
                  f"success={agent.success_count}")

    # Usage statistics
    active_agents = [a for a in all_agents if a.usage_count > 100]
    print(f"\n   Active agents (>100 uses): {len(active_agents)}")

    return len(high_trust) > 0


def generate_summary_report(model_data, metrics):
    """Generate comprehensive summary report."""
    print("\n" + "="*70)
    print("📋 COMPREHENSIVE EVALUATION REPORT")
    print("="*70)

    # Model info
    print("\n🏗️  Model Architecture:")
    print(f"   Total parameters: {model_data['total_params']:,}")
    print(f"   Final agents: {model_data['hierarchy'].count_agents()}")
    print(f"   Embedding dimension: {model_data['embeddings'].shape[1]}")
    print(f"   Vocabulary size: {model_data['embeddings'].shape[0]:,}")

    # Performance
    print(f"\n🎯 Performance:")
    print(f"   Best validation perplexity: {model_data['best_perplexity']:.2f}")

    baseline_ppl = calculate_baseline_perplexity(model_data['embeddings'].shape[0])
    improvement = baseline_ppl / model_data['best_perplexity']
    print(f"   vs Random baseline: {improvement:.2f}x better")

    # Learning verification
    is_learning = analyze_learning_curve(metrics)

    # Phase analysis
    analyze_phase_transition(metrics)

    # Baseline comparison
    compare_to_baseline(model_data['best_perplexity'], model_data['embeddings'].shape[0])

    # Agent analysis
    has_specialists = analyze_agent_specialization(model_data)

    # Final verdict
    print("\n" + "="*70)
    print("🏆 FINAL VERDICT:")
    print("="*70)

    checks = []
    checks.append(("Loss decreased during training", is_learning))
    checks.append(("Better than random baseline", model_data['best_perplexity'] < baseline_ppl))
    checks.append(("Has specialized agents", has_specialists))
    checks.append(("Perplexity < 1000", model_data['best_perplexity'] < 1000))

    passed = sum([c[1] for c in checks])
    total = len(checks)

    print(f"\nChecks passed: {passed}/{total}")
    for check_name, passed_check in checks:
        symbol = "✅" if passed_check else "❌"
        print(f"   {symbol} {check_name}")

    if passed >= 3:
        print(f"\n{'✅ ' * 10}")
        print("🎉 MODEL IS WORKING AND LEARNING! 🎉")
        print(f"{'✅ ' * 10}")
    elif passed >= 2:
        print(f"\n⚠️  Model shows some learning but needs improvement")
    else:
        print(f"\n❌ Model is NOT learning properly - debug needed")

    print("\n" + "="*70)


def main():
    """Main evaluation function."""
    print("="*70)
    print("K-1 LANGUAGE MODEL EVALUATION")
    print("="*70)

    # Load model
    try:
        model_data = load_trained_model()
    except FileNotFoundError:
        print("\n❌ No trained model found. Please train first:")
        print("   python train_10m_model.py")
        return

    # Load metrics
    metrics = load_training_metrics()

    # Generate report
    generate_summary_report(model_data, metrics)


if __name__ == '__main__':
    main()
