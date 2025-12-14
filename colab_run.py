"""
Google Colab training script for Self-Learning K-1 System on WikiText-2.

This script downloads the WikiText-2 dataset, initializes the K-1 system,
and trains it with automatic Phase 1 → Phase 2 transition.

Usage in Google Colab:
    !git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
    %cd self-learning-k-1
    !python colab_run.py
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

print("="*60)
print("Self-Learning K-1 System - WikiText-2 Training")
print("="*60)

# Install dependencies
print("\n📦 Installing dependencies...")
os.system("pip install -q numpy scikit-learn")

# Add k1_system to path
sys.path.insert(0, os.path.abspath('.'))

# Import components
print("\n🔧 Importing K-1 System components...")
from k1_system.data_loader import WikiText2Loader
from k1_system.core import Hierarchy, HierarchicalRouter, TrustSystem
from k1_system.learning import CreditAssignmentSystem, ForwardPass, AdaptiveWeightUpdater
from k1_system.structural import PruningSystem, MergingSystem, GrowingSystem, ReorganizationSystem
from k1_system.autonomy import ParameterController, SystemState, StoppingController, SelfDiagnostic
from k1_system.safety import SnapshotManager, ValidationSystem
from k1_system.initialization import HierarchyBuilder
from k1_system.utils import MetricsTracker, TrainingLogger


class K1TextSystem:
    """
    K-1 System adapted for text/sequence data (WikiText-2).
    """

    def __init__(self, config_path: str = None, embedding_dim: int = 128, vocab_size: int = 10000):
        """
        Initialize K1 System for text.

        Args:
            config_path: Path to configuration file
            embedding_dim: Embedding dimension
            vocab_size: Vocabulary size
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / 'k1_system' / 'config' / 'config_phase1.json'

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Adjust config for shorter training in Colab
        self.config['stopping']['max_iterations'] = 50000  # Shorter for Colab
        self.config['system']['phase_1_duration'] = 5000   # Shorter Phase 1

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Initialize logger
        self.logger = TrainingLogger()
        self.logger.log("Initializing Self-Learning K-1 System for WikiText-2", 'INFO')

        # Initialize hierarchy with language-specific domains
        builder = HierarchyBuilder(
            input_dim=embedding_dim,
            hidden_dim=64,
            output_dim=embedding_dim,
            initial_trust=self.config['trust']['initial']
        )

        # Create language-specific hierarchy
        self.hierarchy = self._build_language_hierarchy(builder)

        # Word embeddings (will be set by data loader)
        self.embeddings = None

        # Initialize all systems (same as original K1System)
        self._initialize_systems()

        # Training state
        self.current_iteration = 0
        self.phase_1_duration = self.config['system']['phase_1_duration']
        self.phase_2_active = False

        self.logger.log("System initialization complete", 'INFO')

    def _build_language_hierarchy(self, builder):
        """Build hierarchy for language modeling."""
        hierarchy = Hierarchy(max_depth=4)

        # Create root
        root = builder._create_agent('master_manager', 'master', 'Language Master')
        hierarchy.set_root(root)

        # Language-specific domains
        domains = [
            ('Syntax', ['Grammar', 'Punctuation', 'Structure']),
            ('Semantics', ['Meaning', 'Context', 'Relations']),
            ('Vocabulary', ['CommonWords', 'RareWords', 'Entities'])
        ]

        for manager_name, agents in domains:
            manager = builder._create_agent(
                f"manager_{manager_name.lower()}",
                'manager',
                manager_name
            )
            hierarchy.add_agent(manager, root)

            for agent_name in agents:
                agent = builder._create_agent(
                    f"agent_{agent_name.lower()}",
                    'agent',
                    agent_name
                )
                hierarchy.add_agent(agent, manager)

        self.logger.log(f"Created language hierarchy with {hierarchy.count_agents()} agents", 'INFO')
        return hierarchy

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
        """Set word embeddings."""
        self.embeddings = embeddings

    def _embed_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Convert sequence of word indices to embeddings."""
        # Average embeddings in sequence
        embedded = self.embeddings[seq]
        return np.mean(embedded, axis=0)

    def train(self, data_loader: WikiText2Loader, num_iterations: int = None):
        """
        Train on WikiText-2 dataset.

        Args:
            data_loader: WikiText2Loader instance
            num_iterations: Number of iterations (None for max from config)
        """
        if num_iterations is None:
            num_iterations = self.config['stopping']['max_iterations']

        # Set embeddings
        self.embeddings = data_loader.get_embeddings(self.embedding_dim)

        self.logger.log(f"Starting training for up to {num_iterations} iterations", 'INFO')
        self.logger.log(f"Phase 1 (Fixed): 0-{self.phase_1_duration}", 'INFO')
        self.logger.log(f"Phase 2 (Autonomous): {self.phase_1_duration}+", 'INFO')

        batch_size = self.config['learning']['batch_size']

        # Training loop
        for iteration in range(num_iterations):
            self.current_iteration = iteration
            self.forward_pass.update_iteration(iteration)

            # PHASE TRANSITION
            if iteration == self.phase_1_duration:
                self._activate_phase_2()

            # Get batch
            batch_x, batch_y = data_loader.get_batch('train', batch_size)

            # Training step
            batch_accuracy, batch_loss = self._training_step(batch_x, batch_y)

            # Validation (every 100 iterations)
            if iteration % 100 == 0:
                val_x, val_y = data_loader.get_batch('val', batch_size)
                val_accuracy = self._validate(val_x, val_y)
            else:
                val_accuracy = batch_accuracy

            # Update metrics
            all_agents = self.hierarchy.get_all_agents()
            avg_trust = self.trust_system.compute_avg_trust(all_agents)

            self.metrics_tracker.update(
                iteration=iteration,
                accuracy=val_accuracy,
                loss=batch_loss,
                avg_trust=avg_trust,
                total_agents=self.hierarchy.count_agents()
            )

            # Phase 2: Parameter adjustment
            if iteration >= self.phase_1_duration:
                self._phase_2_adjustments(iteration, all_agents)

            # Structural operations
            self._run_structural_operations(iteration, val_accuracy)

            # Logging
            if iteration % 100 == 0:
                self._log_progress(iteration, val_accuracy, batch_loss, avg_trust)

            # Stopping criteria
            should_stop, reason = self.stopping_controller.update(iteration, val_accuracy, batch_loss)
            if should_stop:
                self.logger.log(f"Stopping: {reason}", 'INFO')
                break

        self._finalize_training()

    def _training_step(self, batch_x, batch_y):
        """Execute one training step on sequence batch."""
        batch_size = len(batch_x)
        total_loss = 0.0
        correct = 0

        for i in range(batch_size):
            # Embed sequence
            x = self._embed_sequence(batch_x[i])
            target_seq = batch_y[i]
            target = target_seq[0]  # Predict first next word

            # Forward pass
            output, routing_path = self.forward_pass.forward(x, mode='hard')

            # Compute loss
            loss = self.forward_pass.compute_loss(output, np.array([target]), task='classification')
            total_loss += loss

            # Check prediction (using embedding similarity)
            pred_embedding = output
            similarities = self.embeddings @ pred_embedding
            pred_idx = np.argmax(similarities)

            if pred_idx == int(target):
                correct += 1

            # Credit assignment and updates (simplified)
            activated_agents = routing_path.get_activated_agents()
            selected_agents = self.credit_assignment.assign_credit(
                routing_path, loss, np.array([target]), output
            )

            for agent in selected_agents:
                gradient = self.weight_updater.compute_gradient(
                    agent, x, np.array([target]), output
                )
                self.weight_updater.update_agent(agent, gradient)

                # Trust update
                error_reduction = 0.01  # Simplified
                if error_reduction > 0:
                    self.trust_system.report_success(agent, error_reduction)

            # Record patterns
            self.reorganization_system.record_activation_pattern(activated_agents)

        accuracy = correct / batch_size
        avg_loss = total_loss / batch_size

        return accuracy, avg_loss

    def _validate(self, val_x, val_y):
        """Run validation."""
        correct = 0
        for i in range(len(val_x)):
            x = self._embed_sequence(val_x[i])
            output, _ = self.forward_pass.forward(x, mode='hard')

            # Predict using embedding similarity
            similarities = self.embeddings @ output
            pred_idx = np.argmax(similarities)

            if pred_idx == int(val_y[i][0]):
                correct += 1

        return correct / len(val_x)

    def _activate_phase_2(self):
        """Activate Phase 2."""
        self.logger.log_phase_transition(self.current_iteration)
        self.param_controller.activate_phase_2()
        self.phase_2_active = True

    def _phase_2_adjustments(self, iteration, all_agents):
        """Phase 2 parameter adjustments."""
        current_metrics = self.metrics_tracker.get_current_metrics()

        state = SystemState(
            iteration=iteration,
            accuracy=current_metrics.get('accuracy', 0.0),
            loss=current_metrics.get('loss', 1.0),
            validation_accuracy=current_metrics.get('accuracy', 0.0),
            total_agents=self.hierarchy.count_agents(),
            active_agents_pct=len(self.hierarchy.get_active_agents()) / max(1, self.hierarchy.count_agents()),
            avg_trust=self.trust_system.compute_avg_trust(all_agents),
            trust_variance=self.trust_system.compute_trust_variance(all_agents),
            cache_hit_rate=0.5,
            rollback_count_last_10k=self.snapshot_manager.get_rollback_count(10000),
            iterations_since_improvement=self.stopping_controller.iterations_since_improvement,
            structural_ops_time_pct=0.1,
            avg_error_magnitude=current_metrics.get('loss', 0.5),
            improvement_last_5k=0.01,
            structural_changes_last_10k=5,
            cache_size=self.trust_system.trust_cache.get_cache_size()
        )

        self.param_controller.check_and_adjust(iteration, state, all_agents)
        self._apply_adjusted_parameters()

    def _apply_adjusted_parameters(self):
        """Apply adjusted parameters."""
        self.pruning_system.update_thresholds(
            trust_threshold=self.param_controller.deletion_threshold
        )
        self.merging_system.update_threshold(
            self.param_controller.merge_similarity_threshold
        )
        self.credit_assignment.update_top_k(self.param_controller.top_k)

    def _run_structural_operations(self, iteration, current_performance):
        """Run structural operations."""
        if iteration % self.param_controller.prune_interval == 0 and iteration > 0:
            results = self.pruning_system.prune_agents(iteration)
            if results['deleted'] > 0:
                self.logger.log_structural_operation('Pruning', results)

        if iteration % self.param_controller.merge_interval == 0 and iteration > 0:
            results = self.merging_system.merge_agents_batch(
                iteration,
                self.reorganization_system.total_activations
            )
            if results['merged'] > 0:
                self.logger.log_structural_operation('Merging', results)

    def _log_progress(self, iteration, accuracy, loss, avg_trust):
        """Log progress."""
        phase = "Phase 2" if self.phase_2_active else "Phase 1"
        self.logger.log(
            f"[{phase}] Iter {iteration}: Acc={accuracy:.4f}, Loss={loss:.4f}, "
            f"Trust={avg_trust:.3f}, Agents={self.hierarchy.count_agents()}",
            'INFO'
        )

    def _finalize_training(self):
        """Finalize training."""
        self.logger.log("Training complete!", 'INFO')
        final_stats = self.metrics_tracker.compute_performance_summary()
        self.logger.log(f"Final: {final_stats}", 'INFO')
        self.logger.finalize()


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("STARTING WIKITEXT-2 TRAINING")
    print("="*60 + "\n")

    # Load WikiText-2 dataset
    print("📚 Loading WikiText-2 dataset...")
    data_loader = WikiText2Loader(
        data_dir='data',
        vocab_size=10000,
        seq_length=50
    )
    data_loader.load_data()

    # Initialize system
    print("\n🚀 Initializing K-1 System...")
    system = K1TextSystem(
        embedding_dim=128,
        vocab_size=len(data_loader.vocab)
    )

    # Train
    print("\n🎯 Starting training...")
    print("⏱️  Phase 1: Iterations 0-5,000 (Fixed Parameters)")
    print("⏱️  Phase 2: Iterations 5,000+ (Autonomous Adjustment)")
    print("\nTraining will stop automatically when converged.\n")

    system.train(data_loader)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📊 Check logs/ directory for detailed training logs")
    print("📈 Metrics saved to logs/metrics_*.json")


if __name__ == '__main__':
    main()
