"""
K-1 Self-Learning Language Model.

The main model that implements the hierarchical self-learning architecture.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
import pickle

# Import all components
from core.hierarchy import Hierarchy, build_initial_hierarchy
from core.routing import HierarchicalRouter
from core.trust import TrustSystem
from learning.backward import BackpropEngine, softmax, cross_entropy_loss
from learning.forward import ForwardPass
from learning.credit import CreditAssignment
from structural.pruning import PruningSystem
from structural.merging import MergingSystem
from structural.growing import GrowingSystem
from structural.reorganization import ReorganizationSystem
from autonomy.parameter_controller import ParameterController, SystemState
from autonomy.stopping import StoppingController
from autonomy.diagnostic import SelfDiagnostic
from safety.snapshot import SnapshotManager
from utils.metrics import MetricsTracker
from utils.logger import Logger


class K1SelfLearningLM:
    """
    K-1 Self-Learning Language Model.

    Features:
    - Hierarchical agent structure
    - Trust-based credit assignment
    - Sparse updates (only top-K agents)
    - Phase 1: Fixed parameters
    - Phase 2: Autonomous parameter adjustment
    - Self-pruning, merging, growing, reorganization
    """

    def __init__(self, config: Dict):
        """
        Initialize K-1 model.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Model dimensions
        self.vocab_size = config.get('vocab_size', 10000)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 128)

        # Training config
        self.phase_1_duration = config.get('phase_1_duration', 10000)
        self.max_iterations = config.get('max_iterations', 50000)

        # Build hierarchy
        self.hierarchy = build_initial_hierarchy(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            initial_trust=config.get('initial_trust', 0.3),
            domains=config.get('domains', None)
        )

        # Initialize systems
        self._init_systems(config)

        # Embeddings and output projection (will be set during training)
        self.embeddings = None
        self.output_projection = None

        # State
        self.current_iteration = 0
        self.phase_2_active = False
        self.best_perplexity = float('inf')

        # Logger
        self.logger = Logger(name='k1_model')
        self.logger.section("K-1 Self-Learning Model Initialized")
        self.logger.info(f"Hierarchy: {self.hierarchy.count_agents()} agents")

    def _init_systems(self, config: Dict):
        """Initialize all subsystems."""
        # Trust system
        self.trust_system = TrustSystem(
            error_penalty_multiplier=config.get('trust', {}).get('error_penalty', 0.95),
            success_reward_multiplier=config.get('trust', {}).get('success_reward', 0.3),
            success_reward_cap=config.get('trust', {}).get('reward_cap', 0.2),
            cache_threshold=config.get('trust', {}).get('cache_threshold', 0.7)
        )

        # Router
        self.router = HierarchicalRouter(
            hierarchy=self.hierarchy,
            confidence_threshold=config.get('routing', {}).get('confidence_threshold', 0.5),
            max_depth=config.get('routing', {}).get('max_depth', 6),
            exploration_rate=config.get('exploration', {}).get('initial_rate', 0.1)
        )

        # Forward pass
        self.forward_pass = ForwardPass(
            hierarchy=self.hierarchy,
            router=self.router
        )

        # Backprop engine
        self.backprop = BackpropEngine(
            learning_rate=config.get('learning', {}).get('learning_rate', 0.0001),
            gradient_clip=config.get('learning', {}).get('gradient_clip', 1.0),
            use_adam=True
        )

        # Credit assignment
        self.credit = CreditAssignment(
            trust_system=self.trust_system,
            top_k=config.get('learning', {}).get('top_k', 3)
        )

        # Structural systems
        self.pruning = PruningSystem(
            hierarchy=self.hierarchy,
            trust_threshold=config.get('pruning', {}).get('trust_threshold', 0.2),
            usage_threshold=config.get('pruning', {}).get('usage_threshold', 50),
            dormancy_threshold=config.get('pruning', {}).get('dormancy_threshold', 10000)
        )

        self.merging = MergingSystem(
            hierarchy=self.hierarchy,
            similarity_threshold=config.get('merging', {}).get('similarity_threshold', 0.85),
            min_trust=config.get('merging', {}).get('min_trust', 0.3)
        )

        self.growing = GrowingSystem(
            hierarchy=self.hierarchy,
            error_cluster_min_size=config.get('growing', {}).get('min_cluster_size', 100),
            persistence_threshold=config.get('growing', {}).get('persistence', 5000),
            max_agents=config.get('structure', {}).get('max_agents', 100)
        )

        self.reorganization = ReorganizationSystem(
            hierarchy=self.hierarchy,
            max_moves_per_cycle=0.1
        )

        # Autonomy
        self.param_controller = ParameterController(config)
        self.stopping = StoppingController(
            max_iterations=self.max_iterations,
            patience=config.get('stopping', {}).get('patience', 10000),
            min_improvement=config.get('stopping', {}).get('min_improvement', 0.001)
        )
        self.diagnostic = SelfDiagnostic()

        # Safety
        self.snapshots = SnapshotManager(max_snapshots=5)

        # Metrics
        self.metrics = MetricsTracker()

    def set_embeddings(self, embeddings: np.ndarray):
        """Set word embeddings."""
        self.embeddings = embeddings.copy()
        # Output projection: maps hidden to vocab logits
        self.output_projection = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01

    def _forward_step(self, token_idx: int) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Single forward step.

        Args:
            token_idx: Input token index

        Returns:
            (logits, hidden, activated_agents)
        """
        # Embed token
        x = self.embeddings[token_idx]

        # Route through hierarchy
        hidden, path = self.forward_pass.forward(x, mode='hard')

        # Project to vocabulary
        logits = hidden @ self.output_projection

        return logits, hidden, path.get_activated_agents()

    def _training_step(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Single training step with proper backpropagation.

        Args:
            batch_x: Input sequences (batch_size, seq_len)
            batch_y: Target sequences (batch_size, seq_len)

        Returns:
            (average_loss, perplexity)
        """
        batch_size, seq_len = batch_x.shape
        total_loss = 0.0
        total_tokens = 0

        # Process batch
        for i in range(min(batch_size, 8)):  # Process 8 sequences
            for t in range(seq_len):
                # Forward pass
                logits, hidden, activated_agents = self._forward_step(batch_x[i, t])
                target_idx = batch_y[i, t]

                # Compute probabilities and loss
                probs = softmax(logits)
                loss = cross_entropy_loss(probs, target_idx)
                total_loss += loss
                total_tokens += 1

                # Skip if no agents activated
                if not activated_agents:
                    continue

                # === TRUST-BASED SELECTION (K-1 innovation) ===
                selected = self.credit.select_agents_for_update(
                    activated_agents,
                    loss
                )

                if not selected:
                    continue

                # === BACKPROPAGATION (proper gradients) ===
                # Gradient of cross-entropy w.r.t. logits
                d_logits = self.backprop.compute_output_layer_gradient(probs, target_idx)

                # Update output projection
                d_output_proj = np.outer(hidden, d_logits)
                self.output_projection -= self.backprop.learning_rate * np.clip(d_output_proj, -0.1, 0.1)

                # Gradient w.r.t. hidden
                d_hidden = d_logits @ self.output_projection.T

                # Update ONLY selected agents (sparse update)
                x = self.embeddings[batch_x[i, t]]

                for agent, responsibility, ranking in selected:
                    gradient = self.backprop.compute_agent_gradient(agent, x, d_hidden)
                    self.backprop.apply_gradient(agent, gradient)

                # Update embeddings
                self.embeddings[batch_x[i, t]] -= self.backprop.learning_rate * np.clip(d_hidden, -0.1, 0.1)

                # === TRUST UPDATE (based on ACTUAL improvement) ===
                # Re-forward to measure actual improvement
                new_logits, _, _ = self._forward_step(batch_x[i, t])
                new_probs = softmax(new_logits)
                new_loss = cross_entropy_loss(new_probs, target_idx)

                self.credit.update_trust_based_on_improvement(
                    selected, loss, new_loss
                )

                # Record for structural operations
                self.merging.record_coactivation([a.id for a in activated_agents])
                self.reorganization.record_activation_pattern(activated_agents)

                if loss > 1.0:
                    self.growing.record_error(
                        d_hidden, loss, activated_agents, self.current_iteration
                    )

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(min(avg_loss, 20))  # Cap to avoid overflow

        return avg_loss, perplexity

    def _evaluate(self, data_loader, split: str = 'val', num_batches: int = 10) -> float:
        """
        Evaluate model.

        Args:
            data_loader: Data loader
            split: Data split to evaluate
            num_batches: Number of batches

        Returns:
            Perplexity
        """
        total_loss = 0.0
        total_tokens = 0

        for _ in range(num_batches):
            batch_x, batch_y = data_loader.get_batch(split, batch_size=16)
            batch_size, seq_len = batch_x.shape

            for i in range(min(batch_size, 4)):
                for t in range(0, seq_len, 4):  # Sample every 4th token
                    logits, _, _ = self._forward_step(batch_x[i, t])
                    probs = softmax(logits)
                    loss = cross_entropy_loss(probs, batch_y[i, t])

                    if not np.isnan(loss) and loss < 20:
                        total_loss += loss
                        total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        return np.exp(min(avg_loss, 20))

    def train(self, data_loader, num_iterations: int = None):
        """
        Train the model.

        Args:
            data_loader: Data loader
            num_iterations: Number of iterations (default: from config)
        """
        if num_iterations is None:
            num_iterations = self.max_iterations

        # Initialize embeddings
        if self.embeddings is None:
            self.set_embeddings(data_loader.get_embeddings(self.embedding_dim))

        self.logger.section("STARTING TRAINING")
        self.logger.info(f"Phase 1: Iterations 0-{self.phase_1_duration}")
        self.logger.info(f"Phase 2: Iterations {self.phase_1_duration}+")
        self.logger.info(f"Max iterations: {num_iterations}")

        # Initial evaluation
        initial_ppl = self._evaluate(data_loader, 'val')
        self.logger.info(f"Initial perplexity: {initial_ppl:.2f}")
        self.best_perplexity = initial_ppl

        start_time = time.time()

        # Training loop
        for iteration in range(num_iterations):
            self.current_iteration = iteration
            self.forward_pass.update_iteration(iteration)

            # === PHASE TRANSITION ===
            if iteration == self.phase_1_duration:
                self._activate_phase_2()

            # === TRAINING STEP ===
            batch_x, batch_y = data_loader.get_batch('train', batch_size=32)
            train_loss, train_ppl = self._training_step(batch_x, batch_y)

            # === EVALUATION (every 500 iterations) ===
            if iteration % 500 == 0:
                val_ppl = self._evaluate(data_loader, 'val')

                # Record metrics
                self.metrics.record(
                    iteration=iteration,
                    loss=train_loss,
                    perplexity=val_ppl,
                    agent_count=self.hierarchy.count_agents(),
                    avg_trust=self.trust_system.compute_avg_trust(self.hierarchy.get_all_agents())
                )

                # Check for improvement
                if val_ppl < self.best_perplexity:
                    self.best_perplexity = val_ppl
                    self.snapshots.create_snapshot(iteration, self.hierarchy)

                # Log progress
                elapsed = time.time() - start_time
                self._log_progress(iteration, train_ppl, val_ppl, elapsed)

            # === PHASE 2: AUTONOMOUS ADJUSTMENT ===
            if self.phase_2_active and iteration % 1000 == 0:
                self._phase_2_update(iteration, train_loss, train_ppl)

            # === STRUCTURAL OPERATIONS ===
            self._run_structural_ops(iteration)

            # === STOPPING CHECK ===
            should_stop, reason = self.stopping.update(iteration, train_loss, train_ppl)
            if should_stop:
                self.logger.info(f"STOPPING: {reason}")
                break

        # Final evaluation
        self._finalize_training(data_loader)

    def _activate_phase_2(self):
        """Activate Phase 2 autonomous mode."""
        self.logger.section("PHASE 2 ACTIVATED")
        self.param_controller.activate_phase_2()
        self.phase_2_active = True

    def _phase_2_update(self, iteration: int, loss: float, perplexity: float):
        """Phase 2 autonomous updates."""
        # Collect system state
        agents = self.hierarchy.get_all_agents()
        state = SystemState(
            iteration=iteration,
            loss=loss,
            perplexity=perplexity,
            total_agents=len(agents),
            active_agents_pct=0.7,  # Approximate
            avg_trust=self.trust_system.compute_avg_trust(agents),
            trust_variance=self.trust_system.compute_trust_variance(agents),
            cache_hit_rate=self.trust_system.trust_cache.get_hit_rate(),
            rollback_count=self.snapshots.get_rollback_count(),
            iterations_since_improvement=self.stopping.iterations_since_improvement,
            structural_changes=0,
            avg_error_magnitude=loss
        )

        # Autonomous parameter adjustment
        self.param_controller.check_and_adjust(iteration, state)

        # Apply adjusted parameters
        self.credit.update_top_k(self.param_controller.top_k)
        self.backprop.update_learning_rate(self.param_controller.learning_rate)
        self.router.update_exploration_rate(self.param_controller.exploration_rate)
        self.pruning.update_thresholds(
            trust_threshold=self.param_controller.prune_trust_threshold
        )
        self.merging.update_threshold(self.param_controller.merge_similarity_threshold)

        # Run diagnostics
        results = self.diagnostic.run_diagnostics(
            iteration=iteration,
            loss_history=list(self.metrics.losses),
            trust_values=[a.trust for a in agents],
            agent_count=len(agents),
            rollback_count=self.snapshots.get_rollback_count(),
            structural_time_pct=0.1
        )

        for result in results:
            if result.problem_detected:
                self.diagnostic.apply_correction(result, self.param_controller, iteration)

    def _run_structural_ops(self, iteration: int):
        """Run structural operations if scheduled."""
        params = self.param_controller.get_current_params()

        # Pruning
        if iteration > 0 and iteration % params['prune_interval'] == 0:
            result = self.pruning.execute_pruning(
                iteration,
                trust_cache=self.trust_system.trust_cache
            )
            if result['pruned'] > 0:
                self.logger.info(f"Pruned {result['pruned']} agents")

        # Merging
        if iteration > 0 and iteration % params['merge_interval'] == 0:
            result = self.merging.execute_merging_cycle(iteration)
            if result['merged'] > 0:
                self.logger.info(f"Merged {result['merged']} agent pairs")

        # Growing
        if iteration > 5000 and iteration % params['grow_interval'] == 0:
            result = self.growing.execute_growing_cycle(iteration)
            if result['agents_created'] > 0:
                self.logger.info(f"Created {result['agents_created']} new agents")

        # Reorganization
        if iteration > 0 and iteration % params['reorganize_interval'] == 0:
            result = self.reorganization.execute_reorganization(iteration)
            if result['agents_moved'] > 0:
                self.logger.info(f"Reorganized: moved {result['agents_moved']} agents")

    def _log_progress(self, iteration: int, train_ppl: float, val_ppl: float, elapsed: float):
        """Log training progress."""
        phase = "Phase 2" if self.phase_2_active else "Phase 1"
        hours = elapsed / 3600

        self.logger.info(
            f"[{phase}] Iter {iteration:,} | "
            f"Train PPL: {train_ppl:.2f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Best: {self.best_perplexity:.2f} | "
            f"Agents: {self.hierarchy.count_agents()} | "
            f"Time: {hours:.2f}h"
        )

    def _finalize_training(self, data_loader):
        """Finalize training."""
        self.logger.section("TRAINING COMPLETE")

        # Final test evaluation
        test_ppl = self._evaluate(data_loader, 'test')

        self.logger.info(f"Final Results:")
        self.logger.info(f"  Best Val Perplexity: {self.best_perplexity:.2f}")
        self.logger.info(f"  Test Perplexity: {test_ppl:.2f}")
        self.logger.info(f"  Final Agents: {self.hierarchy.count_agents()}")
        self.logger.info(f"  Phase 2 Adjustments: {self.param_controller.get_adjustment_count()}")

        # Print stopping report
        report = self.stopping.generate_report(self.current_iteration, "training complete")
        print(report)

        self.logger.close()

    def save(self, path: str):
        """Save model to file."""
        data = {
            'embeddings': self.embeddings,
            'output_projection': self.output_projection,
            'hierarchy': self.hierarchy,
            'config': self.config,
            'best_perplexity': self.best_perplexity
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embeddings']
        self.output_projection = data['output_projection']
        self.hierarchy = data['hierarchy']
        self.best_perplexity = data.get('best_perplexity', float('inf'))
        print(f"Model loaded from {path}")

    def get_statistics(self) -> Dict:
        """Get model statistics."""
        return {
            'hierarchy': self.hierarchy.get_statistics(),
            'trust': self.trust_system.get_trust_distribution(self.hierarchy.get_all_agents()),
            'backprop': self.backprop.get_statistics(),
            'credit': self.credit.get_statistics(),
            'pruning': self.pruning.get_statistics(),
            'merging': self.merging.get_statistics(),
            'growing': self.growing.get_statistics(),
            'stopping': self.stopping.get_statistics()
        }

    def get_stats(self) -> Dict:
        """Get model stats for run_colab.py compatibility."""
        # Count total parameters
        total_params = 0
        if self.embeddings is not None:
            total_params += self.embeddings.size
        if self.output_projection is not None:
            total_params += self.output_projection.size

        for agent in self.hierarchy.get_all_agents():
            for key in ['W1', 'b1', 'W2', 'b2', 'routing']:
                if key in agent.weights:
                    total_params += agent.weights[key].size

        return {
            'total_parameters': total_params,
            'num_agents': self.hierarchy.count_agents(),
            'avg_trust': self.trust_system.compute_avg_trust(self.hierarchy.get_all_agents()),
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1',
            'best_perplexity': self.best_perplexity,
            'iteration': self.current_iteration
        }

    def get_current_phase(self) -> str:
        """Get current training phase."""
        return 'Phase 2' if self.phase_2_active else 'Phase 1'

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Single training step wrapper for run_colab.py compatibility.

        Args:
            x: Input sequence (seq_len,)
            y: Target sequence (seq_len,)

        Returns:
            Loss value
        """
        self.current_iteration += 1
        self.forward_pass.update_iteration(self.current_iteration)

        # Initialize embeddings if needed
        if self.embeddings is None:
            vocab_size = self.vocab_size
            self.embeddings = np.random.randn(vocab_size, self.embedding_dim) * 0.01
            self.output_projection = np.random.randn(self.embedding_dim, vocab_size) * 0.01

        # Check phase transition
        if self.current_iteration == self.phase_1_duration and not self.phase_2_active:
            self._activate_phase_2()

        # Process as batch of 1
        batch_x = x.reshape(1, -1)
        batch_y = y.reshape(1, -1)

        loss, _ = self._training_step(batch_x, batch_y)

        # Run structural ops periodically
        if self.phase_2_active and self.current_iteration % 1000 == 0:
            self._run_structural_ops(self.current_iteration)

        return loss

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for evaluation.

        Args:
            x: Input sequence (seq_len,)

        Returns:
            Logits (seq_len, vocab_size)
        """
        if self.embeddings is None:
            raise ValueError("Model not initialized. Call train_step first.")

        seq_len = len(x)
        all_logits = []

        for t in range(seq_len):
            logits, _, _ = self._forward_step(x[t])
            all_logits.append(logits)

        return np.array(all_logits)

    def generate(self, prompt: np.ndarray, max_new_tokens: int = 50, temperature: float = 1.0) -> list:
        """
        Generate tokens autoregressively.

        Args:
            prompt: Starting token indices
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated token indices
        """
        if self.embeddings is None:
            raise ValueError("Model not initialized. Call train_step first.")

        generated = list(prompt)

        for _ in range(max_new_tokens):
            # Get last token
            last_token = generated[-1]

            # Forward through hierarchy
            logits, _, _ = self._forward_step(last_token)

            # Apply temperature
            logits = logits / temperature

            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)

            # Sample
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated.append(int(next_token))

        return generated
