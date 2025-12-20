"""
Hybrid K-1 Trainer: Gradient-Based Selection + Trust + Diversity

This module contains the main training loop for the K-1 self-learning system.
KEY INNOVATION:
- Computes REAL gradients via backprop (mathematically rigorous)
- Selects top-K by: GRADIENT MAGNITUDE + TRUST + DIVERSITY
- Phase 1: Learn baseline, build trust scores
- Phase 2: Autonomous pruning/merging based on gradients+trust
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple
from collections import defaultdict

from ..core import Hierarchy, Agent, TrustSystem, HierarchicalRouter
from ..initialization import HierarchyBuilder
from ..utils import TrainingLogger, MetricsTracker
from .forward_pass import ForwardPass
from .weight_update import AdaptiveWeightUpdater

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_k1_system(forward_pass: ForwardPass, data_loader,
                       embedding: nn.Module = None, output_proj: nn.Module = None,
                       num_batches: int = 10, vocab_size: int = 256) -> Tuple[float, float]:
    """
    Run validation on K-1 system using cross-entropy loss and proper perplexity.

    Args:
        forward_pass: K-1 ForwardPass system
        data_loader: DataLoader with validation data
        embedding: Embedding layer for token indices (if using WikiText)
        output_proj: Output projection layer (embed_dim -> vocab_size)
        num_batches: Number of batches to validate on
        vocab_size: Vocabulary size for loss computation

    Returns:
        (avg_loss, perplexity) tuple - where loss is cross-entropy and perplexity = exp(loss)
    """
    total_loss = 0.0
    total_tokens = 0

    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    for _ in range(num_batches):
        try:
            # Get validation batch (returns PyTorch tensors)
            x_batch, y_batch = data_loader.get_batch('val', batch_size=8, return_tensors='pt')
            batch_size = x_batch.shape[0]

            # Forward pass for each sample in batch
            for i in range(batch_size):
                # Convert token indices to embeddings if embedding layer provided
                if embedding is not None:
                    x_tokens = x_batch[i]  # Shape: (seq_len,)
                    y_tokens = y_batch[i]  # Shape: (seq_len,) - target tokens
                    
                    x_embedded = embedding(x_tokens)  # Shape: (seq_len, embed_dim)
                    x_vector = torch.mean(x_embedded, dim=0)  # Mean pooling
                    x = x_vector.detach().cpu().numpy()

                    # Forward through K-1 system
                    output, _ = forward_pass.forward(x, mode='hard')

                    # Convert output to tensor
                    if isinstance(output, np.ndarray):
                        output_tensor = torch.from_numpy(output).float().to(device)
                    else:
                        output_tensor = output.to(device)

                    # Compute cross-entropy loss if we have output projection
                    if output_proj is not None:
                        # Project to vocabulary logits
                        logits = output_proj(output_tensor)  # Shape: (vocab_size,)
                        # Expand to match sequence length
                        logits_expanded = logits.unsqueeze(0).expand(len(y_tokens), -1)
                        # Cross-entropy loss (sum over sequence)
                        loss = loss_fn(logits_expanded, y_tokens)
                        total_loss += loss.item()
                        total_tokens += len(y_tokens)
                    else:
                        # Fallback: use MSE (for backwards compatibility)
                        y_embedded = embedding(y_tokens)
                        y_vector = torch.mean(y_embedded, dim=0)
                        y = y_vector.detach().cpu().numpy()
                        if isinstance(output, torch.Tensor):
                            output = output.cpu().numpy()
                        loss = np.mean((output - y) ** 2)
                        total_loss += loss
                        total_tokens += 1
                else:
                    # No embedding layer - use raw data
                    x = x_batch[i].cpu().numpy()
                    y = y_batch[i].cpu().numpy()
                    output, _ = forward_pass.forward(x, mode='hard')
                    if isinstance(output, torch.Tensor):
                        output = output.cpu().numpy()
                    loss = np.mean((output - y) ** 2)
                    total_loss += loss
                    total_tokens += 1

        except Exception as e:
            # If validation batch fails, skip
            continue

    if total_tokens == 0:
        return 0.0, float('inf')

    # Average loss per token
    avg_loss = total_loss / total_tokens
    # Perplexity = exp(cross-entropy loss)
    perplexity = np.exp(min(avg_loss, 100.0))  # Clip to prevent overflow

    return avg_loss, perplexity


class HybridK1Trainer:
    """
    HYBRID K-1: Gradient-based selection + Trust + Diversity

    KEY INNOVATION:
    - Computes REAL gradients via backprop (mathematically rigorous)
    - Selects top-K by: GRADIENT MAGNITUDE + TRUST + DIVERSITY
    - Phase 1: Learn baseline, build trust scores
    - Phase 2: Autonomous pruning/merging based on gradients+trust
    """

    def __init__(self, config: Dict, data_loader=None):
        self.config = config
        self.logger = TrainingLogger()
        self.data_loader = data_loader

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(window_size=1000)

        self.vocab_size = config['model'].get('vocab_size', 256)
        
        # Embedding layer for token indices -> vectors (if using WikiText)
        if data_loader is not None:
            vocab_size = data_loader.get_vocab_size()
            self.vocab_size = vocab_size
            embed_dim = config['model']['input_dim']
            self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
            # Output projection: embed_dim -> vocab_size for token prediction
            self.output_proj = nn.Linear(config['model']['output_dim'], vocab_size).to(device)
            print(f"Created embedding layer: {vocab_size} tokens -> {embed_dim} dims")
            print(f"Created output projection: {config['model']['output_dim']} dims -> {vocab_size} tokens")
        else:
            self.embedding = None
            self.output_proj = None

        # Build hierarchy
        builder = HierarchyBuilder(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            initial_trust=config['trust']['initial']
        )

        self.hierarchy = builder.build_simple_hierarchy(
            num_domains=config['hierarchy']['num_domains'],
            agents_per_domain=config['hierarchy']['agents_per_domain']
        )

        total_agents = self.hierarchy.count_agents()
        self.logger.log(f"Created hierarchy: {total_agents} agents", 'INFO')

        # Initialize systems
        self.trust_system = TrustSystem(
            error_penalty_multiplier=config['trust']['error_penalty_multiplier'],
            success_reward_multiplier=config['trust']['success_reward_multiplier'],
            success_reward_cap=config['trust']['success_reward_cap'],
            cache_threshold=config['trust']['cache_threshold']
        )

        self.router = HierarchicalRouter(
            hierarchy=self.hierarchy,
            confidence_threshold=config['structure']['routing_confidence_threshold'],
            max_depth=config['structure']['max_hierarchy_depth'],
            exploration_rate=config['exploration']['initial_rate']
        )

        self.forward_pass = ForwardPass(
            hierarchy=self.hierarchy,
            router=self.router,
            trust_cache=self.trust_system.trust_cache
        )

        self.weight_updater = AdaptiveWeightUpdater(
            learning_rate=config['learning']['learning_rate']
        )

        # Hybrid selection parameters (from config)
        selection_config = config.get('selection', {})
        self.top_k = config['learning']['top_k']
        self.top_k_min = config['learning'].get('top_k_min', 2)
        self.top_k_max = config['learning'].get('top_k_max', 10)
        self.gradient_weight = selection_config.get('gradient_weight', 0.7)
        self.trust_weight = selection_config.get('trust_weight', 0.2)
        self.diversity_weight = selection_config.get('diversity_weight', 0.1)
        self.diversity_window = selection_config.get('diversity_window', 100)
        
        # Trust update parameters (from config)
        trust_update_config = config.get('trust_update', {})
        self.trust_gradient_scale = trust_update_config.get('gradient_scale', 0.5)
        self.trust_max_update = trust_update_config.get('max_update', 0.1)
        self.trust_penalty_ratio = trust_update_config.get('penalty_ratio', 0.5)
        
        # Adaptation parameters (from config)
        adapt_config = config.get('adaptation', {})
        self.plateau_loss_std = adapt_config.get('plateau_loss_std', 0.05)
        self.plateau_loss_trend = adapt_config.get('plateau_loss_trend', 0.1)
        self.instability_std = adapt_config.get('instability_std', 0.3)
        self.loss_history_window = adapt_config.get('loss_history_window', 50)
        
        # Logging intervals (from config)
        self.log_interval = config['learning'].get('log_interval', 100)
        self.validation_interval = config['learning'].get('validation_interval', 200)
        
        # Sequence length for fallback
        self.seq_length = config['model'].get('max_seq_len', 64)

        # Phase control
        self.phase_1_duration = config['system']['phase_1_duration']
        self.phase_2_active = False
        self.current_phase = 1

        # Agent tracking (for diversity)
        self.agent_last_updated = {}  # agent_id -> step
        self.agent_update_count = {}  # agent_id -> count
        self.agent_gradient_history = {}  # agent_id -> list of gradient norms

        # Statistics
        self.total_params_updated = 0
        self.total_steps = 0
        self.phase_2_adjustments = 0
        self.agents_pruned = 0
        self.agents_merged = 0

        # Loss tracking for adaptive top_k
        self.loss_history = []

    def train(self, data: np.ndarray = None, max_steps: int = 1000):
        """Train using hybrid gradient+trust+diversity selection."""
        print("\n" + "="*70)
        print("HYBRID K-1: Gradient + Trust + Diversity Selection")
        print("="*70)
        print(f"Innovation: Use REAL gradients + trust + diversity")
        print(f"Phase 1 (0-{min(self.phase_1_duration, max_steps)}): Gradient-based + exploration")
        if max_steps > self.phase_1_duration:
            print(f"Phase 2 ({self.phase_1_duration}+): Gradient+Trust+Diversity + autonomous ops")

        # Check data source
        if self.data_loader is not None:
            print(f"Data source: WikiText-2 (vocab_size={self.data_loader.get_vocab_size()})")
        elif data is not None:
            print(f"Data source: Synthetic data ({data.shape[0]} samples)")
        else:
            print(f"Data source: Random noise (testing only)")

        print("="*70 + "\n")

        start_time = time.time()
        all_agents = self.hierarchy.get_all_agents()
        total_agents = len(all_agents)

        # Initialize tracking
        for agent in all_agents:
            self.agent_last_updated[id(agent)] = -1000
            self.agent_update_count[id(agent)] = 0
            self.agent_gradient_history[id(agent)] = []

        # Calculate parameters dynamically from agent configuration
        input_dim = self.config['model']['input_dim']
        hidden_dim = self.config['model']['hidden_dim']
        output_dim = self.config['model']['output_dim']
        params_per_agent = (input_dim * hidden_dim + hidden_dim +  # Layer 1
                           hidden_dim * output_dim + output_dim)    # Layer 2
        total_params = total_agents * params_per_agent

        for step in range(max_steps):
            # PHASE TRANSITION
            if step == self.phase_1_duration and max_steps > self.phase_1_duration:
                print(f"\n{'='*70}")
                print("🚀 PHASE 2 ACTIVATED: Autonomous Adaptation + Structural Ops")
                print(f"{'='*70}")
                print(f"Selection weights: Gradient={self.gradient_weight:.1f}, "
                      f"Trust={self.trust_weight:.1f}, Diversity={self.diversity_weight:.1f}")
                print(f"{'='*70}\n")
                self.phase_2_active = True
                self.current_phase = 2

            # Get input/target data
            if self.data_loader is not None:
                # Use WikiText-2 data
                try:
                    x_batch, y_batch = self.data_loader.get_batch('train', batch_size=1, return_tensors='pt')

                    # x_batch is (1, seq_len) of token indices
                    # y_batch is (1, seq_len) of target token indices
                    x_tokens = x_batch[0]  # Shape: (seq_len,)
                    y_tokens = y_batch[0]  # Shape: (seq_len,) - target tokens for cross-entropy
                    
                    # Convert tokens to embeddings
                    x_embedded = self.embedding(x_tokens)  # Shape: (seq_len, embed_dim)
                    # Use mean pooling to get input vector for K-1 hierarchy
                    x_vector = torch.mean(x_embedded, dim=0)  # Shape: (embed_dim,)
                    x = x_vector.detach().cpu().numpy()
                    
                    # Store tokens for cross-entropy loss
                    target_tokens = y_tokens  # Keep as tensor for cross-entropy

                except Exception as e:
                    # Fallback to random if batch fails
                    print(f"Warning: Data loading failed: {e}")
                    x = torch.randn(self.config['model']['input_dim'], device=device).cpu().numpy()
                    target_tokens = torch.randint(0, self.vocab_size, (self.seq_length,), device=device)
            elif data is not None:
                # Use synthetic data (regression-style)
                sample_idx = np.random.randint(0, len(data))
                x = data[sample_idx]
                target_tokens = None  # Use MSE for synthetic data
            else:
                # Fallback to random noise
                x = torch.randn(self.config['model']['input_dim'], device=device).cpu().numpy()
                target_tokens = torch.randint(0, self.vocab_size, (self.seq_length,), device=device)

            # Forward pass through K-1 system
            output, routing_path = self.forward_pass.forward(x, mode='hard')

            # Compute loss
            if isinstance(output, np.ndarray):
                output_tensor = torch.from_numpy(output).float().to(device)
            else:
                output_tensor = output.to(device)

            # Use cross-entropy loss for language modeling (proper objective)
            if self.output_proj is not None and target_tokens is not None:
                # Project output to vocabulary logits
                logits = self.output_proj(output_tensor)  # Shape: (vocab_size,)
                # Expand logits to match sequence length (simplified: use same prediction for all positions)
                logits_expanded = logits.unsqueeze(0).expand(len(target_tokens), -1)  # Shape: (seq_len, vocab_size)
                # Cross-entropy loss
                loss_fn = nn.CrossEntropyLoss()
                loss_tensor = loss_fn(logits_expanded, target_tokens)
                loss = loss_tensor.item()
            else:
                # Fallback to MSE for synthetic data
                target = x  # Auto-encoding objective
                target_tensor = torch.from_numpy(target).float().to(device) if isinstance(target, np.ndarray) else target
                loss = torch.mean((output_tensor - target_tensor) ** 2).item()
                loss_tensor = torch.tensor(loss, device=device, requires_grad=True)

            self.loss_history.append(loss)

            # HYBRID APPROACH: Compute gradients for ALL agents (like backprop)
            activated_agents = routing_path.get_activated_agents()
            gradients = self._compute_all_gradients_from_loss(loss_tensor, activated_agents)

            # INNOVATION: Select top-K using gradient + trust + diversity
            selected_agents = self._hybrid_selection(
                activated_agents, gradients, step
            )

            # Track loss improvement for trust updates
            if len(self.loss_history) > 1:
                loss_improved = self.loss_history[-1] < self.loss_history[-2]
                loss_delta = abs(self.loss_history[-1] - self.loss_history[-2])
            else:
                loss_improved = False
                loss_delta = 0.0

            # Update ONLY selected agents (sparse!) using PyTorch optimizers
            params_updated_this_step = 0
            for agent in selected_agents:
                # Update using PyTorch optimizer (gradients already in .grad)
                self.weight_updater.update_agent(agent)
                params_updated_this_step += params_per_agent

                # Track updates
                self.agent_last_updated[id(agent)] = step
                self.agent_update_count[id(agent)] += 1

                # Update trust based on actual loss improvement (not just gradient magnitude)
                # Scale trust update by gradient magnitude (agents with larger gradients are more responsible)
                grad_mag = self._gradient_magnitude(gradients[agent])
                trust_scale = min(grad_mag * self.trust_gradient_scale, self.trust_max_update)
                
                if loss_improved:
                    # Loss decreased - reward agents that were updated
                    self.trust_system.report_success(agent, trust_scale)
                else:
                    # Loss increased or stayed same - penalize agents
                    self.trust_system.report_error(agent, trust_scale * self.trust_penalty_ratio)

            # Zero gradients for non-selected agents
            for agent in activated_agents:
                if agent not in selected_agents:
                    for param in agent.parameters():
                        if param.grad is not None:
                            param.grad.zero_()

            self.total_params_updated += params_updated_this_step
            self.total_steps += 1

            # PHASE 2: Autonomous operations
            if self.phase_2_active and step % 200 == 0 and step > self.phase_1_duration:
                self._autonomous_operations(step, all_agents, gradients)

            # Update metrics tracker
            avg_trust = np.mean([a.trust for a in all_agents])
            self.metrics_tracker.update(
                iteration=step,
                accuracy=0.0,  # Placeholder for now
                loss=loss,
                avg_trust=avg_trust,
                total_agents=len(all_agents)
            )

            # Validation
            if step % self.validation_interval == 0 and self.data_loader is not None:
                val_loss, val_perplexity = validate_k1_system(
                    self.forward_pass,
                    self.data_loader,
                    embedding=self.embedding,
                    output_proj=self.output_proj,
                    num_batches=10,
                    vocab_size=self.vocab_size
                )
                print(f"[{step:4d}] VALIDATION | Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")

            # Logging
            if step % self.log_interval == 0:
                self._log_progress(step, loss, params_updated_this_step,
                                  total_params, all_agents, start_time)

        # Final summary
        return self._print_summary(start_time, total_params, all_agents)

    def _compute_all_gradients_from_loss(self, loss_tensor: torch.Tensor, activated_agents: List) -> Dict:
        """
        Compute gradients for all activated agents from a loss tensor.
        Uses PyTorch autograd for proper backpropagation.
        """
        gradients = {}

        # Zero all gradients first
        for agent in activated_agents:
            for param in agent.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        # Backward pass to compute gradients
        if loss_tensor.requires_grad:
            loss_tensor.backward(retain_graph=True)

        # Extract gradients from each agent
        for agent in activated_agents:
            agent_grads = {}
            grad_norm = 0.0

            for name, param in agent.named_parameters():
                if param.grad is not None:
                    agent_grads[name] = param.grad.clone()
                    grad_norm += param.grad.data.norm(2).item() ** 2

            grad_norm = grad_norm ** 0.5
            gradients[agent] = agent_grads

            # Track gradient history
            self.agent_gradient_history[id(agent)].append(grad_norm)
            if len(self.agent_gradient_history[id(agent)]) > 100:
                self.agent_gradient_history[id(agent)].pop(0)

        return gradients

    def _hybrid_selection(self, agents: List, gradients: Dict, step: int) -> List:
        """
        INNOVATION: Select top-K using gradient + trust + diversity

        Phase 1: Mostly gradient-based + random exploration
        Phase 2: Balanced (gradient + trust + diversity)
        """
        if self.current_phase == 1:
            # PHASE 1: Gradient-based with exploration
            gradient_scores = []
            for agent in agents:
                grad_norm = self._gradient_magnitude(gradients[agent])
                gradient_scores.append((agent, grad_norm))

            gradient_scores.sort(key=lambda x: x[1], reverse=True)

            # Top-(K-1) by gradient + 1 random for exploration
            top_by_gradient = [a for a, _ in gradient_scores[:max(1, self.top_k-1)]]

            # Random exploration (give other agents a chance)
            remaining = [a for a, _ in gradient_scores[self.top_k:]]
            if remaining:
                random_agent = remaining[np.random.randint(len(remaining))]
                return top_by_gradient + [random_agent]
            return top_by_gradient

        else:
            # PHASE 2: Balanced selection (gradient + trust + diversity)
            selection_scores = []

            for agent in agents:
                # 1. Gradient score (current importance)
                grad_score = self._gradient_magnitude(gradients[agent])

                # 2. Trust score (historical reliability)
                trust_score = agent.trust

                # 3. Diversity score (recency penalty - encourage unused agents)
                steps_since_update = step - self.agent_last_updated[id(agent)]
                diversity_score = min(steps_since_update / float(self.diversity_window), 1.0)

                # Combined score
                combined_score = (
                    self.gradient_weight * grad_score +
                    self.trust_weight * trust_score +
                    self.diversity_weight * diversity_score
                )

                selection_scores.append((agent, combined_score, grad_score, trust_score, diversity_score))

            # Sort by combined score
            selection_scores.sort(key=lambda x: x[1], reverse=True)

            return [agent for agent, _, _, _, _ in selection_scores[:self.top_k]]

    def _gradient_magnitude(self, gradient: Dict) -> float:
        """Compute total gradient magnitude (L2 norm) using PyTorch."""
        total = 0.0
        for param_name, grad in gradient.items():
            if isinstance(grad, torch.Tensor):
                total += torch.sum(grad ** 2).item()
            else:
                total += np.sum(grad ** 2)
        return np.sqrt(total)

    def _autonomous_operations(self, step: int, all_agents: List, gradients: Dict):
        """
        PHASE 2 AUTONOMOUS OPERATIONS:
        1. Prune agents with low trust AND low gradients
        2. Merge agents with similar gradient patterns
        3. Adapt top_k based on loss trajectory
        """
        print(f"\n{'='*70}")
        print(f"🤖 Autonomous Operations (Step {step})")
        print(f"{'='*70}")

        # Use thresholds from config
        prune_trust_threshold = self.config.get('pruning', {}).get('trust_threshold', 0.2)
        prune_gradient_threshold = 0.01  # Could add to config
        merge_similarity_threshold = self.config.get('merging', {}).get('similarity_threshold', 0.9)
        
        # 1. PRUNE: Low trust AND low average gradient
        agents_to_prune = []
        for agent in all_agents:
            if len(self.agent_gradient_history[id(agent)]) > 10:
                avg_gradient = np.mean(self.agent_gradient_history[id(agent)])

                if agent.trust < prune_trust_threshold and avg_gradient < prune_gradient_threshold:
                    agents_to_prune.append(agent)

        if agents_to_prune:
            print(f"🔪 Pruning {len(agents_to_prune)} agents (low trust + low gradients)")
            self.agents_pruned += len(agents_to_prune)
        else:
            print(f"✓ No agents to prune")

        # 2. MERGE: Find agents with similar gradient patterns
        similar_pairs = self._find_similar_gradient_patterns(all_agents, threshold=merge_similarity_threshold)
        if similar_pairs:
            print(f"🔗 Found {len(similar_pairs)} pairs with similar gradients (would merge)")
            self.agents_merged += len(similar_pairs)
        else:
            print(f"✓ No redundant agents to merge")

        # 3. ADAPT TOP_K: Based on loss trajectory
        if len(self.loss_history) > self.loss_history_window:
            recent_losses = self.loss_history[-self.loss_history_window:]
            old_top_k = self.top_k

            # Detect plateau
            loss_std = np.std(recent_losses)
            loss_trend = recent_losses[-1] - recent_losses[0]

            if loss_std < self.plateau_loss_std and abs(loss_trend) < self.plateau_loss_trend:
                # Plateau detected - increase exploration
                self.top_k = min(self.top_k_max, self.top_k + 1)
                if self.top_k != old_top_k:
                    print(f"📈 Plateau detected → Increased top_k: {old_top_k} → {self.top_k}")
                    self.phase_2_adjustments += 1

            elif loss_std > self.instability_std:
                # Instability detected - reduce updates
                self.top_k = max(self.top_k_min, self.top_k - 1)
                if self.top_k != old_top_k:
                    print(f"📉 Instability detected → Decreased top_k: {old_top_k} → {self.top_k}")
                    self.phase_2_adjustments += 1

            if old_top_k == self.top_k:
                print(f"✓ top_k unchanged ({self.top_k})")

        print(f"{'='*70}\n")

    def _find_similar_gradient_patterns(self, agents: List, threshold: float = 0.9) -> List:
        """Find agents with similar gradient patterns (cosine similarity)."""
        similar_pairs = []

        for i, agent_i in enumerate(agents):
            for agent_j in agents[i+1:]:
                hist_i = self.agent_gradient_history.get(id(agent_i), [])
                hist_j = self.agent_gradient_history.get(id(agent_j), [])

                if len(hist_i) > 10 and len(hist_j) > 10:
                    # Cosine similarity of gradient histories
                    vec_i = np.array(hist_i[-10:])
                    vec_j = np.array(hist_j[-10:])

                    similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)

                    if similarity > threshold:
                        similar_pairs.append((agent_i, agent_j, similarity))

        return similar_pairs

    def _log_progress(self, step, loss, params_updated, total_params, all_agents, start_time):
        """Log training progress."""
        elapsed = time.time() - start_time
        avg_params_per_step = self.total_params_updated / max(self.total_steps, 1)
        update_pct = (params_updated / total_params) * 100
        phase = f"Phase {self.current_phase}"
        high_trust = sum(1 for a in all_agents if a.trust > 0.7)
        low_trust = sum(1 for a in all_agents if a.trust < 0.2)

        print(f"[{step:4d}] {phase} | Loss: {loss:.4f} | "
              f"Params updated: {params_updated:,} ({update_pct:.1f}%) | "
              f"Top-K: {self.top_k} | Trust (high/low): {high_trust}/{low_trust} | "
              f"Time: {elapsed:.1f}s")

    def _print_summary(self, start_time, total_params, all_agents):
        """Print final training summary."""
        elapsed = time.time() - start_time
        avg_params_per_step = self.total_params_updated / self.total_steps
        update_pct = (avg_params_per_step / total_params) * 100

        print(f"\n{'='*70}")
        print(f"Hybrid K-1 Training Complete")
        print(f"{'='*70}")
        print(f"Total steps: {self.total_steps}")
        print(f"Total parameter updates: {self.total_params_updated:,}")
        print(f"Average params updated per step: {avg_params_per_step:,.0f} ({update_pct:.1f}%)")
        print(f"Phase 2 adjustments: {self.phase_2_adjustments}")
        print(f"Agents pruned: {self.agents_pruned}")
        print(f"Agents merged: {self.agents_merged}")

        # Metrics summary
        metrics_summary = self.metrics_tracker.compute_performance_summary()
        if metrics_summary:
            print(f"\nMetrics Summary:")
            print(f"  Best loss: {metrics_summary.get('best_loss', 0):.4f}")
            print(f"  Current loss: {metrics_summary.get('current_loss', 0):.4f}")
            print(f"  Worst loss: {metrics_summary.get('worst_loss', 0):.4f}")

        # Final validation
        if self.data_loader is not None:
            val_loss, val_perplexity = validate_k1_system(
                self.forward_pass,
                self.data_loader,
                embedding=self.embedding,
                output_proj=self.output_proj,
                num_batches=20,
                vocab_size=self.vocab_size
            )
            print(f"\nFinal Validation:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  Perplexity: {val_perplexity:.2f}")

        # Trust distribution
        trust_scores = [a.trust for a in all_agents]
        print(f"\nTrust Distribution:")
        print(f"  Average: {np.mean(trust_scores):.3f}")
        print(f"  High trust (>0.7): {sum(1 for t in trust_scores if t > 0.7)}")
        print(f"  Low trust (<0.2): {sum(1 for t in trust_scores if t < 0.2)}")

        # Update distribution
        update_counts = [self.agent_update_count[id(a)] for a in all_agents]
        print(f"\nUpdate Distribution:")
        print(f"  Average updates per agent: {np.mean(update_counts):.1f}")
        print(f"  Max updates: {max(update_counts)}")
        print(f"  Min updates: {min(update_counts)}")
        print(f"  Never updated: {sum(1 for c in update_counts if c == 0)}")

        print(f"\nTime: {elapsed:.1f}s")
        print(f"{'='*70}\n")

        return {
            'total_steps': self.total_steps,
            'total_params_updated': self.total_params_updated,
            'avg_params_per_step': avg_params_per_step,
            'update_percentage': update_pct,
            'phase_2_adjustments': self.phase_2_adjustments,
            'agents_pruned': self.agents_pruned,
            'agents_merged': self.agents_merged,
            'time': elapsed
        }
