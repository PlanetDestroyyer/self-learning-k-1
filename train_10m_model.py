"""
Training script for 50M parameter K-1 Language Model.

This script trains a properly scaled K-1 system on WikiText-2 dataset
with comprehensive evaluation to verify learning.
"""

import sys
import os
import json
import numpy as np
import time
from pathlib import Path

print("="*70)
print(" 50M PARAMETER SELF-LEARNING K-1 LANGUAGE MODEL")
print("="*70)

# Add to path
sys.path.insert(0, os.path.abspath('.'))

# Install dependencies
print("\n📦 Installing dependencies...")
os.system("pip install -q numpy scikit-learn tqdm 2>/dev/null")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Import K-1 components
print("🔧 Importing K-1 System...")
from k1_system.llm_data_loader import LLMDataLoader
from k1_system.core import Hierarchy, HierarchicalRouter, TrustSystem
from k1_system.learning import CreditAssignmentSystem, ForwardPass, AdaptiveWeightUpdater
from k1_system.structural import PruningSystem, MergingSystem, GrowingSystem, ReorganizationSystem
from k1_system.autonomy import ParameterController, SystemState, StoppingController, SelfDiagnostic
from k1_system.safety import SnapshotManager
from k1_system.initialization import HierarchyBuilder
from k1_system.utils import MetricsTracker, TrainingLogger


class K1LanguageModel:
    """
    50M parameter K-1 Language Model with proper evaluation.

    Architecture:
    - Vocabulary: 30,000 words (WikiText-2 full vocab)
    - Embedding dimension: 384
    - Hierarchy: 90 agents (384→384→384)
    - Total parameters: ~52M
    """

    def __init__(self, config_path: str = None):
        """Initialize 50M parameter model."""
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / 'k1_system' / 'config' / 'config_phase1.json'

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Model configuration (50M parameters)
        self.vocab_size = 30000  # Full WikiText vocab
        self.embedding_dim = 384
        self.hidden_dim = 384
        self.seq_length = 128
        self.num_agents_target = 90

        # Adjust training config
        self.config['stopping']['max_iterations'] = 100000
        self.config['system']['phase_1_duration'] = 10000

        # Initialize logger
        self.logger = TrainingLogger()
        self.logger.log("Initializing 50M Parameter K-1 Language Model", 'INFO')

        # Build hierarchy
        self.hierarchy = self._build_scaled_hierarchy()

        # Word embeddings (will be set by data loader)
        self.embeddings = None
        self.output_projection = None  # vocab_size x embedding_dim

        # Count parameters
        self._count_parameters()

        # Initialize all systems
        self._initialize_systems()

        # Training state
        self.current_iteration = 0
        self.phase_1_duration = self.config['system']['phase_1_duration']
        self.phase_2_active = False

        # Evaluation metrics
        self.best_perplexity = float('inf')
        self.perplexity_history = []

        self.logger.log(f"Model initialized: {self.total_params:,} parameters", 'INFO')

    def _build_scaled_hierarchy(self):
        """Build hierarchy with ~90 agents for 50M params."""
        hierarchy = Hierarchy(max_depth=4)

        builder = HierarchyBuilder(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            initial_trust=self.config['trust']['initial']
        )

        # Root
        root = builder._create_agent('master', 'master', 'Language Model')
        hierarchy.set_root(root)

        # 5 domain managers with more agents each for 50M model
        domains = [
            ('Syntax', 18),      # 18 agents
            ('Semantics', 18),   # 18 agents
            ('Vocabulary', 18),  # 18 agents
            ('Context', 18),     # 18 agents
            ('Structure', 18)    # 18 agents
        ]
        # Total: 5 managers + 90 agents + 1 root = 96 agents

        for manager_name, num_agents in domains:
            manager = builder._create_agent(
                f"mgr_{manager_name.lower()}",
                'manager',
                manager_name
            )
            hierarchy.add_agent(manager, root)

            # Add agents under each manager
            for i in range(num_agents):
                agent = builder._create_agent(
                    f"agent_{manager_name.lower()}_{i}",
                    'agent',
                    f"{manager_name}_{i}"
                )
                hierarchy.add_agent(agent, manager)

        self.logger.log(f"Created hierarchy: {hierarchy.count_agents()} agents", 'INFO')
        return hierarchy

    def _count_parameters(self):
        """Count total parameters in model."""
        # Embeddings: vocab_size x embedding_dim
        embedding_params = self.vocab_size * self.embedding_dim

        # Agents: each has W1, b1, W2, b2, routing weights
        params_per_agent = (
            self.embedding_dim * self.hidden_dim +  # W1
            self.hidden_dim +                        # b1
            self.hidden_dim * self.embedding_dim +   # W2
            self.embedding_dim +                     # b2
            self.embedding_dim * 10                  # routing (approximate)
        )

        num_agents = self.hierarchy.count_agents()
        agent_params = num_agents * params_per_agent

        # Output projection: embedding_dim x vocab_size
        output_params = self.embedding_dim * self.vocab_size

        self.total_params = embedding_params + agent_params + output_params

        self.logger.log(f"\nParameter Breakdown:", 'INFO')
        self.logger.log(f"  Embeddings: {embedding_params:,} ({embedding_params/1e6:.2f}M)", 'INFO')
        self.logger.log(f"  Agents ({num_agents}): {agent_params:,} ({agent_params/1e6:.2f}M)", 'INFO')
        self.logger.log(f"  Output: {output_params:,} ({output_params/1e6:.2f}M)", 'INFO')
        self.logger.log(f"  TOTAL: {self.total_params:,} ({self.total_params/1e6:.2f}M)", 'INFO')

    def _initialize_systems(self):
        """Initialize all subsystems."""
        # Core systems
        self.trust_system = TrustSystem(
            error_penalty_multiplier=self.config['trust']['error_penalty_multiplier'],
            success_reward_multiplier=self.config['trust']['success_reward_multiplier'],
            success_reward_cap=self.config['trust']['success_reward_cap'],
            cache_threshold=self.config['trust']['cache_threshold']
        )

        self.router = HierarchicalRouter(
            hierarchy=self.hierarchy,
            confidence_threshold=self.config['structure']['routing_confidence_threshold'],
            max_depth=self.config['structure']['max_hierarchy_depth'],
            exploration_rate=self.config['exploration']['initial_rate']
        )

        self.forward_pass = ForwardPass(
            hierarchy=self.hierarchy,
            router=self.router,
            trust_cache=self.trust_system.trust_cache
        )

        self.weight_updater = AdaptiveWeightUpdater(
            learning_rate=self.config['learning']['learning_rate']
        )

        self.credit_assignment = CreditAssignmentSystem(
            trust_system=self.trust_system,
            top_k=self.config['learning']['top_k']
        )

        # Structural operations
        self.pruning_system = PruningSystem(
            hierarchy=self.hierarchy,
            trust_threshold=self.config['pruning']['trust_threshold'],
            usage_threshold=self.config['pruning']['usage_threshold'],
            dormancy_threshold=self.config['pruning']['dormancy_threshold'],
            validation_drop_threshold=self.config['pruning']['validation_drop_threshold'],
            min_agents_per_manager=self.config['structure']['min_agents_per_manager']
        )

        self.merging_system = MergingSystem(
            hierarchy=self.hierarchy,
            similarity_threshold=self.config['merging']['similarity_threshold'],
            min_trust=self.config['merging']['min_trust'],
            validation_drop_threshold=self.config['merging']['validation_drop_threshold']
        )

        self.growing_system = GrowingSystem(
            hierarchy=self.hierarchy,
            error_cluster_min_size=self.config['growing']['error_cluster_min_size'],
            error_frequency_threshold=self.config['growing']['error_frequency_threshold'],
            new_agent_initial_trust=self.config['growing']['new_agent_initial_trust'],
            validation_period=self.config['growing']['validation_period'],
            success_threshold=self.config['growing']['success_threshold'],
            failure_threshold=self.config['growing']['failure_threshold'],
            max_agents=self.config['structure']['max_total_agents']
        )

        self.reorganization_system = ReorganizationSystem(
            hierarchy=self.hierarchy,
            max_moves_per_cycle=0.1
        )

        # Autonomy
        self.param_controller = ParameterController(self.config)
        self.stopping_controller = StoppingController(
            max_iterations=self.config['stopping']['max_iterations'],
            patience=self.config['stopping']['patience'],
            min_improvement=self.config['stopping']['min_improvement']
        )
        self.diagnostic_system = SelfDiagnostic()

        # Safety
        self.snapshot_manager = SnapshotManager(
            max_snapshots=self.config['rollback']['max_snapshots'],
            performance_drop_threshold=self.config['rollback']['performance_drop_threshold']
        )

        # Metrics
        self.metrics_tracker = MetricsTracker(window_size=1000)

    def set_embeddings(self, embeddings: np.ndarray):
        """Set word embeddings and initialize output projection."""
        self.embeddings = embeddings
        # Output projection: map embeddings to vocab logits
        self.output_projection = np.random.randn(self.embedding_dim, self.vocab_size) * 0.02

    def train(self, data_loader: LLMDataLoader, num_iterations: int = None):
        """
        Train the 50M parameter model.

        Args:
            data_loader: Data loader
            num_iterations: Number of iterations
        """
        if num_iterations is None:
            num_iterations = self.config['stopping']['max_iterations']

        # Set embeddings and initialize output projection
        embeddings = data_loader.get_embeddings(self.embedding_dim)
        self.set_embeddings(embeddings)

        self.logger.log(f"\n{'='*70}", 'INFO')
        self.logger.log(f"STARTING TRAINING - 50M PARAMETER K-1 MODEL", 'INFO')
        self.logger.log(f"{'='*70}", 'INFO')
        self.logger.log(f"Dataset: {data_loader.dataset_name}", 'INFO')
        self.logger.log(f"Vocabulary: {len(data_loader.vocab):,} words", 'INFO')
        self.logger.log(f"Training sequences: {len(data_loader.train_data):,}", 'INFO')
        self.logger.log(f"Model parameters: {self.total_params:,}", 'INFO')
        self.logger.log(f"Phase 1 (Fixed): 0-{self.phase_1_duration:,}", 'INFO')
        self.logger.log(f"Phase 2 (Autonomous): {self.phase_1_duration:,}+", 'INFO')
        self.logger.log(f"{'='*70}\n", 'INFO')

        batch_size = self.config['learning']['batch_size']
        start_time = time.time()

        # Initial evaluation
        initial_ppl = self._evaluate(data_loader, 'val')
        self.logger.log(f"Initial validation perplexity: {initial_ppl:.2f}", 'INFO')
        self.best_perplexity = initial_ppl

        # Training loop with progress bar
        iterator = range(num_iterations)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Training")

        for iteration in iterator:
            self.current_iteration = iteration
            self.forward_pass.update_iteration(iteration)

            # PHASE TRANSITION
            if iteration == self.phase_1_duration:
                self._activate_phase_2()

            # Training step
            batch_x, batch_y = data_loader.get_batch('train', batch_size)
            train_loss, train_ppl = self._training_step(batch_x, batch_y, data_loader)

            # Validation (every 500 iterations)
            if iteration % 500 == 0:
                val_ppl = self._evaluate(data_loader, 'val')
                self.perplexity_history.append((iteration, val_ppl))

                # Check for improvement
                if val_ppl < self.best_perplexity:
                    improvement = self.best_perplexity - val_ppl
                    self.best_perplexity = val_ppl
                    self.logger.log(
                        f"🎯 New best perplexity: {val_ppl:.2f} (↓{improvement:.2f})",
                        'INFO'
                    )

                # Log progress
                elapsed = time.time() - start_time
                self._log_progress(iteration, train_ppl, val_ppl, elapsed)

            # Phase 2: Parameter adjustment
            if iteration >= self.phase_1_duration and iteration % 1000 == 0:
                self._phase_2_adjustments(iteration)

            # Structural operations
            if iteration % 5000 == 0 and iteration > 0:
                self._run_structural_operations(iteration)

            # Update progress bar
            if HAS_TQDM:
                iterator.set_postfix({
                    'train_ppl': f'{train_ppl:.2f}',
                    'best_val_ppl': f'{self.best_perplexity:.2f}',
                    'agents': self.hierarchy.count_agents()
                })

            # Stopping criteria
            should_stop, reason = self.stopping_controller.update(
                iteration, 1.0 / train_ppl, train_loss
            )
            if should_stop:
                self.logger.log(f"\n🛑 Stopping training: {reason}", 'INFO')
                break

        # Final evaluation
        self._finalize_training(data_loader)

    def _training_step(self, batch_x, batch_y, data_loader):
        """Execute one training step."""
        batch_size, seq_len = batch_x.shape
        total_loss = 0.0
        total_log_prob = 0.0
        total_tokens = 0

        for i in range(batch_size):
            # Process sequence
            seq_loss = 0.0
            seq_log_prob = 0.0

            for t in range(seq_len):
                # Embed current token
                current_token_idx = batch_x[i, t]
                x = self.embeddings[current_token_idx]

                # Forward pass through hierarchy
                hidden, routing_path = self.forward_pass.forward(x, mode='hard')

                # Project to vocabulary
                logits = hidden @ self.output_projection

                # Target token
                target_idx = batch_y[i, t]

                # Compute loss (cross-entropy)
                logits_max = np.max(logits)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / np.sum(exp_logits)

                # Clip for numerical stability
                prob_target = np.clip(probs[target_idx], 1e-10, 1.0)
                loss = -np.log(prob_target)

                seq_loss += loss
                seq_log_prob += np.log(prob_target)

                # Gradient and update (simplified)
                if t % 10 == 0:  # Update every 10 tokens
                    activated_agents = routing_path.get_activated_agents()
                    if activated_agents:
                        selected_agents = self.credit_assignment.assign_credit(
                            routing_path, loss, np.array([target_idx]), hidden
                        )

                        for agent in selected_agents:
                            gradient = self.weight_updater.compute_gradient(
                                agent, x, np.array([target_idx]), hidden
                            )
                            self.weight_updater.update_agent(agent, gradient)

                            # Trust update
                            error_reduction = 0.001
                            if error_reduction > 0:
                                self.trust_system.report_success(agent, error_reduction)

            total_loss += seq_loss / seq_len
            total_log_prob += seq_log_prob / seq_len
            total_tokens += seq_len

        avg_loss = total_loss / batch_size
        perplexity = np.exp(-total_log_prob / total_tokens)

        return avg_loss, perplexity

    def _evaluate(self, data_loader: LLMDataLoader, split: str = 'val') -> float:
        """
        Evaluate model on validation/test set.

        Returns:
            Perplexity
        """
        num_batches = 10  # Evaluate on 10 batches
        total_log_prob = 0.0
        total_tokens = 0

        for _ in range(num_batches):
            batch_x, batch_y = data_loader.get_batch(split, batch_size=32)
            batch_size, seq_len = batch_x.shape

            for i in range(batch_size):
                for t in range(seq_len):
                    # Embed and forward
                    x = self.embeddings[batch_x[i, t]]
                    hidden, _ = self.forward_pass.forward(x, mode='hard')

                    # Project to vocab
                    logits = hidden @ self.output_projection
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)

                    # Log probability of target
                    target_idx = batch_y[i, t]
                    prob_target = np.clip(probs[target_idx], 1e-10, 1.0)
                    total_log_prob += np.log(prob_target)
                    total_tokens += 1

        perplexity = np.exp(-total_log_prob / total_tokens)
        return perplexity

    def _activate_phase_2(self):
        """Activate Phase 2."""
        self.logger.log(f"\n{'='*70}", 'INFO')
        self.logger.log(f"🚀 PHASE 2 ACTIVATED - AUTONOMOUS OPTIMIZATION", 'INFO')
        self.logger.log(f"{'='*70}\n", 'INFO')
        self.param_controller.activate_phase_2()
        self.phase_2_active = True

    def _phase_2_adjustments(self, iteration):
        """Phase 2 parameter adjustments."""
        all_agents = self.hierarchy.get_all_agents()
        current_metrics = self.metrics_tracker.get_current_metrics()

        state = SystemState(
            iteration=iteration,
            accuracy=0.5,  # Placeholder
            loss=current_metrics.get('loss', 1.0),
            validation_accuracy=0.5,
            total_agents=self.hierarchy.count_agents(),
            active_agents_pct=0.7,
            avg_trust=self.trust_system.compute_avg_trust(all_agents),
            trust_variance=self.trust_system.compute_trust_variance(all_agents),
            cache_hit_rate=0.5,
            rollback_count_last_10k=0,
            iterations_since_improvement=self.stopping_controller.iterations_since_improvement,
            structural_ops_time_pct=0.1,
            avg_error_magnitude=0.5,
            improvement_last_5k=0.01,
            structural_changes_last_10k=5,
            cache_size=self.trust_system.trust_cache.get_cache_size()
        )

        self.param_controller.check_and_adjust(iteration, state, all_agents)
        self._apply_adjusted_parameters()

    def _apply_adjusted_parameters(self):
        """Apply adjusted parameters from controller."""
        self.credit_assignment.update_top_k(self.param_controller.top_k)
        self.weight_updater.update_learning_rate(self.param_controller.learning_rate)

    def _run_structural_operations(self, iteration):
        """Run structural operations."""
        results = self.pruning_system.prune_agents(iteration)
        if results.get('deleted', 0) > 0:
            self.logger.log(f"Pruned {results['deleted']} agents", 'INFO')

    def _log_progress(self, iteration, train_ppl, val_ppl, elapsed):
        """Log training progress."""
        phase = "Phase 2" if self.phase_2_active else "Phase 1"
        hours = elapsed / 3600

        self.logger.log(
            f"[{phase}] Iter {iteration:,} | "
            f"Train PPL: {train_ppl:.2f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Best: {self.best_perplexity:.2f} | "
            f"Agents: {self.hierarchy.count_agents()} | "
            f"Time: {hours:.2f}h",
            'INFO'
        )

    def _finalize_training(self, data_loader):
        """Finalize training and evaluate."""
        self.logger.log(f"\n{'='*70}", 'INFO')
        self.logger.log(f"✅ TRAINING COMPLETE", 'INFO')
        self.logger.log(f"{'='*70}\n", 'INFO')

        # Final test evaluation
        test_ppl = self._evaluate(data_loader, 'test')

        self.logger.log(f"Final Results:", 'INFO')
        self.logger.log(f"  Best validation perplexity: {self.best_perplexity:.2f}", 'INFO')
        self.logger.log(f"  Test perplexity: {test_ppl:.2f}", 'INFO')
        self.logger.log(f"  Final agents: {self.hierarchy.count_agents()}", 'INFO')

        # Save model
        self._save_model()

        self.logger.finalize()

    def _save_model(self):
        """Save trained model."""
        import pickle

        save_path = Path('trained_k1_50m.pkl')
        model_data = {
            'hierarchy': self.hierarchy,
            'embeddings': self.embeddings,
            'output_projection': self.output_projection,
            'best_perplexity': self.best_perplexity,
            'total_params': self.total_params
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.log(f"\n💾 Model saved to {save_path}", 'INFO')


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("TRAINING 50M PARAMETER K-1 LANGUAGE MODEL")
    print("="*70 + "\n")

    # Load dataset
    print("📚 Loading dataset...")
    dataset_choice = os.environ.get('K1_DATASET', 'wikitext')  # Use WikiText for richer dataset

    data_loader = LLMDataLoader(
        dataset_name=dataset_choice,
        data_dir='data',
        vocab_size=30000,  # Match model vocab size (50M model)
        seq_length=128,
        train_split=0.9
    )
    data_loader.load_data()

    # Initialize model
    print("\n🚀 Initializing 50M parameter model...")
    model = K1LanguageModel()

    # Train
    print("\n🎯 Starting training...\n")
    model.train(data_loader)

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\n📊 Check logs/ directory for detailed metrics")
    print("💾 Model saved to trained_k1_50m.pkl")


if __name__ == '__main__':
    main()
