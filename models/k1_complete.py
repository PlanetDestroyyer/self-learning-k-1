"""
K-1 Self-Learning System - COMPLETE IMPLEMENTATION
Trust-based hierarchical learning with hybrid LOO attribution

Features:
- Trust = "already learned, skip updates" cooldown mechanism
- Hierarchy = error type matching (managers=domains, specialists=subtopics)
- Hybrid attribution: gradient-based (fast) + LOO verification (accurate)
- Trust decay: good agents stay high, bad agents fade to 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import time


class Agent(nn.Module):
    """Agent with trust-based cooldown and LOO attribution."""
    
    def __init__(self, agent_id: str, domain: str, specialty: str,
                 input_dim: int, hidden_dim: int, output_dim: int,
                 device='cuda'):
        super().__init__()
        
        self.id = agent_id
        self.domain = domain
        self.specialty = specialty
        self.device = device
        
        # Network
        self.ln1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Trust system
        self.trust = 0.0
        self.times_updated = 0
        self.times_skipped = 0
        
        # LOO attribution
        self.loo_contribution = 0.0  # Last computed LOO score
        self.loo_step = 0  # When LOO was last computed
        self.gradient_contribution = 0.0  # Fast gradient-based estimate
        
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        hidden = F.gelu(self.fc1(x_norm))
        hidden_norm = self.ln2(hidden)
        output = self.fc2(hidden_norm)
        
        if self.input_dim == self.output_dim:
            output = output + x
        return output
    
    def decay_trust(self, decay_factor: float = 0.995):
        self.trust *= decay_factor


class K1CompleteSystem(nn.Module):
    """
    K-1 System with hybrid LOO attribution.
    
    Credit Assignment:
    - Fast: Gradient magnitude (every step)
    - Accurate: Leave-One-Out (every N steps)
    
    Trust System:
    - Low trust → update (needs learning)
    - High trust → skip (already learned)
    - After update → trust increases
    - Slow decay → allows re-learning
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dimensions
        self.vocab_size = config.get('vocab_size', 1000)
        self.embed_dim = config.get('embed_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.max_seq_len = config.get('max_seq_len', 64)
        
        # Core layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size)
        
        # Build hierarchy
        self.agents = nn.ModuleList()
        self.managers = []
        self.specialists = {}
        self._build_hierarchy()
        
        # Training params
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.top_k = config.get('top_k', 5)
        self.trust_threshold = config.get('trust_threshold', 0.7)
        self.trust_increase = config.get('trust_increase', 0.1)
        self.trust_decay = config.get('trust_decay', 0.995)
        
        # LOO attribution settings
        self.loo_interval = config.get('loo_interval', 50)  # Compute LOO every N steps
        
        # Phase tracking
        self.current_iteration = 0
        self.phase_1_duration = config.get('phase_1_duration', 10000)
        self.phase_2_active = False
        
        self.to(self.device)
        
    def _build_hierarchy(self):
        """Build: Root → Managers (domains) → Specialists (subtopics)"""
        root = Agent('root', 'Master', 'Root', 
                     self.embed_dim, self.hidden_dim, self.embed_dim, self.device)
        self.agents.append(root)
        
        domains = ['Language', 'Logic', 'Pattern', 'Context']
        for i, domain in enumerate(domains):
            manager = Agent(f'mgr_{domain}', domain, 'Manager',
                           self.embed_dim, self.hidden_dim, self.embed_dim, self.device)
            self.agents.append(manager)
            self.managers.append(manager)
            self.specialists[domain] = []
            
            subtopics = ['Basic', 'Advanced', 'Edge', 'Complex']
            for j, subtopic in enumerate(subtopics):
                spec = Agent(f'spec_{domain}_{subtopic}', domain, subtopic,
                            self.embed_dim, self.hidden_dim, self.embed_dim, self.device)
                self.agents.append(spec)
                self.specialists[domain].append(spec)
        
        print(f"Built hierarchy: 1 root + {len(self.managers)} managers + "
              f"{sum(len(v) for v in self.specialists.values())} specialists = {len(self.agents)} agents")
    
    def forward(self, x: torch.Tensor, exclude_agent: Agent = None) -> torch.Tensor:
        """Forward pass. If exclude_agent is set, skip that agent (for LOO)."""
        B, T = x.shape
        hidden = self.embedding(x)
        flat = hidden.reshape(B * T, self.embed_dim)
        
        # Normalized trust-weighted contributions
        active_agents = [a for a in self.agents if a != exclude_agent]
        total_weight = sum(0.1 + a.trust for a in active_agents)
        
        agent_sum = torch.zeros_like(flat)
        for agent in active_agents:
            agent_out = agent.forward(flat)
            weight = (0.1 + agent.trust) / total_weight
            agent_sum = agent_sum + agent_out * weight
        
        flat = flat + agent_sum * 0.1
        hidden = flat.reshape(B, T, self.embed_dim)
        logits = self.output_proj(hidden)
        
        return logits
    
    def compute_loo_contributions(self, x: torch.Tensor, y: torch.Tensor, 
                                   base_loss: float) -> Dict[str, float]:
        """
        Compute Leave-One-Out contribution for each agent.
        Positive = agent made loss WORSE (blame it)
        Negative = agent made loss BETTER (credit it)
        """
        contributions = {}
        
        with torch.no_grad():
            for agent in self.agents:
                # Forward without this agent
                logits_without = self.forward(x, exclude_agent=agent)
                loss_without = F.cross_entropy(
                    logits_without.reshape(-1, self.vocab_size), 
                    y.reshape(-1)
                ).item()
                
                # Contribution = how much loss INCREASES when we REMOVE agent
                # Positive = agent was HELPING (removing it makes loss worse)
                # Negative = agent was HURTING (removing it makes loss better)
                contribution = loss_without - base_loss
                contributions[agent.id] = contribution
                agent.loo_contribution = contribution
                agent.loo_step = self.current_iteration
        
        return contributions
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """Training with hybrid LOO attribution."""
        self.current_iteration += 1
        
        if self.current_iteration == self.phase_1_duration:
            print("\n" + "="*70)
            print("🚀 PHASE 2 ACTIVATED: Self-Learning Mode Enabled")
            print("="*70 + "\n")
            self.phase_2_active = True
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward + loss
        logits = self.forward(x)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        loss_val = loss.item()
        
        # Backprop for gradients
        loss.backward()
        
        # Update embedding and output layers
        with torch.no_grad():
            for param in self.embedding.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad
            for param in self.output_proj.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad
        
        # === HYBRID ATTRIBUTION ===
        # Fast: Gradient-based (every step)
        for agent in self.agents:
            agent.gradient_contribution = sum(
                p.grad.abs().mean().item() for p in agent.parameters() if p.grad is not None
            )
        
        # Accurate: LOO verification (every N steps)
        loo_computed = False
        if self.current_iteration % self.loo_interval == 0:
            loo_contributions = self.compute_loo_contributions(x, y, loss_val)
            loo_computed = True
        
        # Rank agents by combined score
        agent_scores = []
        for agent in self.agents:
            # Use gradient contribution, weighted by recent LOO if available
            score = agent.gradient_contribution
            if agent.loo_step > 0:
                # Positive LOO = agent was helping, negative = hurting
                # We want to UPDATE agents that are HURTING (negative LOO)
                loo_factor = 1.0 - agent.loo_contribution  # Invert: hurting agents get higher score
                score *= loo_factor
            agent_scores.append((agent, score))
        
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-K, but SKIP high-trust agents
        candidates = []
        skipped = 0
        for agent, score in agent_scores:
            if agent.trust < self.trust_threshold:
                candidates.append((agent, score))
            else:
                agent.times_skipped += 1
                skipped += 1
            if len(candidates) >= self.top_k:
                break
        
        # Update selected agents
        updated = 0
        with torch.no_grad():
            for agent, _ in candidates:
                for param in agent.parameters():
                    if param.grad is not None:
                        param.data -= self.learning_rate * param.grad
                agent.trust = min(1.0, agent.trust + self.trust_increase)
                agent.times_updated += 1
                updated += 1
        
        # Zero gradients
        self.zero_grad()
        
        # Decay trust
        for agent in self.agents:
            agent.decay_trust(self.trust_decay)
        
        return {
            'loss': loss_val,
            'updated': updated,
            'skipped': skipped,
            'total_agents': len(self.agents),
            'avg_trust': np.mean([a.trust for a in self.agents]),
            'high_trust': sum(1 for a in self.agents if a.trust > self.trust_threshold),
            'loo_computed': loo_computed,
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1'
        }
    
    def get_stats(self) -> Dict:
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'total_agents': len(self.agents),
            'avg_trust': np.mean([a.trust for a in self.agents]),
            'high_trust_agents': sum(1 for a in self.agents if a.trust > self.trust_threshold),
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1',
            'iteration': self.current_iteration
        }
    
    def get_agent_status(self) -> List[Dict]:
        return [{
            'id': a.id,
            'domain': a.domain,
            'specialty': a.specialty,
            'trust': round(a.trust, 3),
            'updated': a.times_updated,
            'skipped': a.times_skipped,
            'loo_score': round(a.loo_contribution, 4),
            'grad_score': round(a.gradient_contribution, 4)
        } for a in self.agents]
    
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0) -> List[int]:
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
        
        self.train()
        return tokens


def create_k1_complete_model(config: Dict) -> K1CompleteSystem:
    return K1CompleteSystem(config)
