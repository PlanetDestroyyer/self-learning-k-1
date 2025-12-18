"""
Complete K-1 Self-Learning System Implementation
Trust-based hierarchical learning with sparse updates and self-evolution

Based on original K-1 plan with PyTorch GPU acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


class Agent(nn.Module):
    """
    Individual agent in the hierarchy with trust-based learning.
    Each agent is a specialist that can be updated independently.
    """
    
    def __init__(self, agent_id: str, agent_type: str, specialty: str,
                 input_dim: int, hidden_dim: int, output_dim: int,
                 num_children: int = 4, device='cuda'):
        super().__init__()
        
        # Identity
        self.id = agent_id
        self.type = agent_type  # 'manager', 'agent', 'sub_agent'
        self.specialty = specialty
        self.device = device
        
        # Network architecture with layer norm and residuals
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Agent's neural network (with modern architecture)
        self.ln1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Router for child selection (soft routing)
        self.num_children = num_children
        self.router = nn.Linear(hidden_dim, num_children)
        
        # Move to device
        self.to(device)
        
        # Trust system
        self.trust = 0.3  # Initial trust
        self.trust_history = []
        
        # Activity tracking
        self.activation_level = 0.0
        self.was_active_this_iteration = False
        self.usage_count = 0
        self.last_used_iteration = 0
        
        # For local gradient updates
        self.last_input = None
        self.last_output = None
        self.last_hidden = None
        
        # Performance tracking
        self.contribution_history = []
        self.success_count = 0
        self.failure_count = 0
        
        # Hierarchy
        self.parent = None
        self.children = []
        
        # Metadata
        self.creation_iteration = 0
        self.protected = False
        
    def forward(self, x: torch.Tensor, return_routing=False):
        """
        Forward pass through this agent.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            return_routing: If True, also return routing logits
            
        Returns:
            output: Agent's output (batch_size, output_dim)
            routing_logits: Optional routing scores for children
        """
        self.was_active_this_iteration = True
        self.last_input = x.detach()
        
        # Layer 1 with LayerNorm and GELU
        x_norm = self.ln1(x)
        hidden = F.gelu(self.fc1(x_norm))
        self.last_hidden = hidden.detach()
        
        # Layer 2 with LayerNorm and residual
        hidden_norm = self.ln2(hidden)
        output = self.fc2(hidden_norm)
        
        # Residual connection (if dims match)
        if self.input_dim == self.output_dim:
            output = output + x
        
        self.last_output = output.detach()
        self.activation_level = output.abs().mean().item()
        
        if return_routing:
            routing_logits = self.router(hidden)
            return output, routing_logits
        
        return output
    
    def update_weights_local(self, error: torch.Tensor, learning_rate: float, 
                            contribution_score: float):
        """
        Hybrid: Local gradients + contribution weighting.
        
        This gives you:
        - Efficient learning (gradients work!)
        - Sparse updates (only top-K agents)
        - Interpretability (know who learned what)
        - Trust-based credit assignment
        """
        if self.last_input is None or self.last_output is None:
            return
        
        # Enable gradients for this agent's parameters only
        for param in self.parameters():
            param.requires_grad = True
        
        # Forward through this agent only
        output = self.forward(self.last_input.requires_grad_(True))
        
        # Loss for this agent's contribution (weighted by trust and contribution)
        # Higher trust and contribution = stronger update signal
        weight = contribution_score * self.trust
        agent_loss = (error.detach() * output).mean() * weight
        
        # Backward ONLY through this agent
        agent_loss.backward()
        
        # Update this agent's weights
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
        
        # Track contribution
        self.contribution_history.append(contribution_score)
        if len(self.contribution_history) > 1000:
            self.contribution_history.pop(0)
    
    def update_trust(self, improvement: float, penalty: bool = False):
        """
        Update trust based on performance.
        
        Args:
            improvement: How much this agent helped (positive) or hurt (negative)
            penalty: If True, apply penalty for error
        """
        if penalty:
            # Penalize for contributing to error
            self.trust *= 0.9
            self.failure_count += 1
        else:
            # Reward for helping reduce error
            reward = min(improvement * 0.5, 0.3)
            self.trust = min(1.0, self.trust + reward)
            self.success_count += 1
        
        self.trust = max(0.0, self.trust)  # Clamp to [0, 1]
        self.trust_history.append(self.trust)
        
        if len(self.trust_history) > 10000:
            self.trust_history.pop(0)
    
    def add_child(self, child):
        """Add child agent."""
        self.children.append(child)
        child.parent = self
    
    def remove_child(self, child):
        """Remove child agent."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None


class Hierarchy(nn.Module):
    """
    Manages the hierarchical structure of agents.
    Implements soft routing and multi-level forward pass.
    """
    
    def __init__(self, root_agent: Agent):
        super().__init__()
        self.root = root_agent
        self.all_agents = {}
        self._register_agent_recursive(root_agent)
        
        # Make all agents accessible as module
        self.agent_modules = nn.ModuleDict({
            agent_id: agent for agent_id, agent in self.all_agents.items()
        })
    
    def _register_agent_recursive(self, agent: Agent):
        """Recursively register all agents."""
        self.all_agents[agent.id] = agent
        for child in agent.children:
            self._register_agent_recursive(child)
    
    def forward(self, x: torch.Tensor, max_depth: int = 3, 
                routing_mode: str = 'soft') -> Tuple[torch.Tensor, List[Agent]]:
        """
        Hierarchical forward pass with soft or hard routing.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            max_depth: Maximum depth to traverse
            routing_mode: 'soft' (weighted combination) or 'hard' (best child)
            
        Returns:
            output: Final output tensor
            active_agents: List of agents that were active
        """
        batch_size = x.shape[0]
        current_hidden = x
        active_agents = []
        
        # Start at root
        current_agent = self.root
        
        for depth in range(max_depth + 1):
            # Process current agent
            agent_output, routing_logits = current_agent.forward(
                current_hidden, return_routing=True
            )
            active_agents.append(current_agent)
            
            # Accumulate output (residual style)
            current_hidden = current_hidden + agent_output
            
            # Route to children if not at max depth
            if depth < max_depth and len(current_agent.children) > 0:
                if routing_mode == 'soft':
                    # Soft routing: weighted combination of children
                    routing_weights = F.softmax(routing_logits, dim=1)
                    
                    # Get outputs from all children
                    child_outputs = []
                    for child in current_agent.children:
                        child_out, _ = child.forward(current_hidden, return_routing=True)
                        child_outputs.append(child_out)
                        active_agents.append(child)
                    
                    # Weighted combination
                    if child_outputs:
                        stacked = torch.stack(child_outputs, dim=1)  # (B, num_children, output_dim)
                        weights_expanded = routing_weights.unsqueeze(-1)  # (B, num_children, 1)
                        child_contribution = (stacked * weights_expanded).sum(dim=1)
                        current_hidden = current_hidden + child_contribution
                    
                    # For next iteration, pick best child
                    best_child_idx = routing_weights.argmax(dim=1)[0].item()
                    if best_child_idx < len(current_agent.children):
                        current_agent = current_agent.children[best_child_idx]
                    else:
                        break
                        
                else:  # hard routing
                    # Hard routing: pick best child
                    best_child_idx = routing_logits.argmax(dim=1)[0].item()
                    if best_child_idx < len(current_agent.children):
                        current_agent = current_agent.children[best_child_idx]
                        active_agents.append(current_agent)
                    else:
                        break
            else:
                break
        
        return current_hidden, active_agents


class TrustSystem:
    """
    Manages trust scores and credit assignment.
    """
    
    def __init__(self):
        self.trust_cache = {}  # High-trust agents
        self.error_history = defaultdict(list)
        
    def compute_contributions(self, active_agents: List[Agent], 
                            error: torch.Tensor) -> List[Tuple[Agent, float]]:
        """
        Compute contribution scores for each active agent.
        
        Args:
            active_agents: Agents that were active in forward pass
            error: Current error tensor
            
        Returns:
            List of (agent, contribution_score) tuples
        """
        contributions = []
        error_magnitude = error.abs().mean().item()
        
        for agent in active_agents:
            # Contribution based on activation level and error
            responsibility = agent.activation_level * error_magnitude
            
            # Ranking score: responsibility weighted by trust
            ranking_score = responsibility * (1.0 + agent.trust)
            
            contributions.append((agent, ranking_score))
        
        return contributions
    
    def select_top_k(self, contributions: List[Tuple[Agent, float]], 
                    k: int) -> List[Tuple[Agent, float]]:
        """
        Select top-K agents for update.
        
        Args:
            contributions: List of (agent, score) tuples
            k: Number of agents to select
            
        Returns:
            Top-K agents with normalized contribution scores
        """
        # Sort by score
        sorted_contributions = sorted(contributions, key=lambda x: x[1], reverse=True)
        
        # Select top-K
        top_k = sorted_contributions[:k]
        
        # Normalize scores
        if top_k:
            max_score = max(score for _, score in top_k)
            if max_score > 0:
                top_k = [(agent, score / max_score) for agent, score in top_k]
        
        return top_k
    
    def update_cache(self, agent: Agent, threshold: float = 0.7):
        """Add high-trust agents to cache."""
        if agent.trust > threshold:
            self.trust_cache[agent.id] = {
                'agent': agent,
                'trust': agent.trust,
                'specialty': agent.specialty,
                'success_count': agent.success_count,
                'last_updated': time.time()
            }


class K1CompleteSystem(nn.Module):
    """
    Complete K-1 Self-Learning System with:
    - Trust-based credit assignment
    - Sparse top-K updates
    - Hierarchical routing
    - Self-evolution capabilities
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model dimensions
        self.vocab_size = config.get('vocab_size', 1000)
        self.embed_dim = config.get('embed_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.max_seq_len = config.get('max_seq_len', 64)
        
        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size)
        
        # Build hierarchy
        self.hierarchy = self._build_initial_hierarchy()
        
        # Trust system
        self.trust_system = TrustSystem()
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.top_k = config.get('top_k', 3)
        
        # Phase tracking
        self.current_iteration = 0
        self.phase_1_duration = config.get('phase_1_duration', 10000)
        self.phase_2_active = False
        
        # Move to device
        self.to(self.device)
        
    def _build_initial_hierarchy(self) -> Hierarchy:
        """Build initial hierarchical structure."""
        # Root manager
        root = Agent(
            agent_id='root',
            agent_type='manager',
            specialty='Master',
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embed_dim,
            num_children=4,
            device=self.device
        )
        
        # Level 1: Domain managers
        level1_specs = ['Language', 'Logic', 'Pattern', 'Context']
        for i, spec in enumerate(level1_specs):
            manager = Agent(
                agent_id=f'manager_{i}',
                agent_type='manager',
                specialty=spec,
                input_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.embed_dim,
                num_children=4,
                device=self.device
            )
            root.add_child(manager)
            
            # Level 2: Specialist agents
            for j in range(4):
                agent = Agent(
                    agent_id=f'agent_{i}_{j}',
                    agent_type='agent',
                    specialty=f'{spec}_specialist_{j}',
                    input_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.embed_dim,
                    num_children=0,
                    device=self.device
                )
                manager.add_child(agent)
        
        return Hierarchy(root)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through K-1 system.
        
        Args:
            x: Input token indices (batch_size, seq_len)
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        B, T = x.shape
        
        # Embed
        x_emb = self.embedding(x)  # (B, T, embed_dim)
        
        # Process each position through hierarchy
        outputs = []
        for t in range(T):
            pos_input = x_emb[:, t, :]  # (B, embed_dim)
            pos_output, _ = self.hierarchy(pos_input, max_depth=2, routing_mode='soft')
            outputs.append(pos_output)
        
        # Stack outputs
        hidden = torch.stack(outputs, dim=1)  # (B, T, embed_dim)
        
        # Project to vocabulary
        logits = self.output_proj(hidden)
        
        return logits
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Single training step with sparse trust-based updates.
        
        Args:
            x: Input tokens (batch_size, seq_len)
            y: Target tokens (batch_size, seq_len)
            
        Returns:
            Dictionary with loss and metrics
        """
        self.current_iteration += 1
        
        # Check phase transition
        if self.current_iteration == self.phase_1_duration:
            print("\n" + "="*70)
            print("🚀 PHASE 2 ACTIVATED: Self-Learning Mode Enabled")
            print("="*70 + "\n")
            self.phase_2_active = True
        
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward pass (tracking active agents)
        B, T = x.shape
        x_emb = self.embedding(x)
        
        all_active_agents = []
        outputs = []
        
        for t in range(T):
            pos_input = x_emb[:, t, :]
            pos_output, active_agents = self.hierarchy(
                pos_input, max_depth=2, routing_mode='soft'
            )
            outputs.append(pos_output)
            all_active_agents.extend(active_agents)
        
        hidden = torch.stack(outputs, dim=1)
        logits = self.output_proj(hidden)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            y.reshape(-1)
        )
        
        # Compute contributions
        error = logits - F.one_hot(y, self.vocab_size).float()
        contributions = self.trust_system.compute_contributions(
            all_active_agents, error
        )
        
        # Select top-K agents for update
        top_k_agents = self.trust_system.select_top_k(contributions, self.top_k)
        
        # Update only selected agents (SPARSE UPDATES!)
        for agent, contribution_score in top_k_agents:
            agent.update_weights_local(
                error=error,
                learning_rate=self.learning_rate,
                contribution_score=contribution_score
            )
        
        # Measure improvement and update trust
        with torch.no_grad():
            new_logits = self.forward(x)
            new_loss = F.cross_entropy(
                new_logits.reshape(-1, self.vocab_size),
                y.reshape(-1)
            )
            improvement = (loss - new_loss).item()
        
        # Update trust scores
        for agent, contribution_score in top_k_agents:
            if improvement > 0:
                agent.update_trust(improvement, penalty=False)
            else:
                agent.update_trust(0, penalty=True)
        
        # Update trust cache
        for agent, _ in top_k_agents:
            self.trust_system.update_cache(agent)
        
        return {
            'loss': loss.item(),
            'improvement': improvement,
            'num_updated_agents': len(top_k_agents),
            'total_active_agents': len(set(all_active_agents)),
            'avg_trust': np.mean([a.trust for a, _ in top_k_agents]),
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1'
        }
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        all_agents = list(self.hierarchy.all_agents.values())
        
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'total_agents': len(all_agents),
            'high_trust_agents': len(self.trust_system.trust_cache),
            'avg_trust': np.mean([a.trust for a in all_agents]),
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1',
            'iteration': self.current_iteration
        }
    
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0) -> List[int]:
        """Generate text autoregressively."""
        self.eval()
        
        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt)
        
        tokens = prompt.tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = tokens[-self.max_seq_len:]
                x = torch.tensor([context], dtype=torch.long, device=self.device)
                
                logits = self.forward(x)
                next_logits = logits[0, -1] / temperature
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token)
        
        return tokens


def create_k1_complete_model(config: Dict) -> K1CompleteSystem:
    """Factory function to create complete K-1 model."""
    return K1CompleteSystem(config)
