"""
K-1 Self-Learning System - SPARSE INTERPRETABLE UPDATES
Find which agent caused the error → Update ONLY that agent

This is NOT standard backprop (update all).
This is NOT arbitrary top-K selection.
This is: Find responsible agent → Update it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


class Agent(nn.Module):
    """Agent with tracking for error attribution."""
    
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
        
        # Error tracking - which errors does this agent handle?
        self.error_responsibility = 0.0  # Current step's responsibility
        self.total_updates = 0
        self.total_errors_handled = 0.0
        
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        hidden = F.gelu(self.fc1(x_norm))
        hidden_norm = self.ln2(hidden)
        output = self.fc2(hidden_norm)
        
        if self.input_dim == self.output_dim:
            output = output + x
        return output


class K1CompleteSystem(nn.Module):
    """
    K-1 System - Sparse Interpretable Updates
    
    Key Idea:
    1. All agents contribute to forward pass
    2. On error, find which agent CAUSED it (gradient attribution)
    3. Update ONLY that agent (not all like backprop)
    4. Track which agent learns what (interpretability)
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
        
        # Core layers (these ALWAYS update - they're shared infrastructure)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size)
        
        # Build agent hierarchy
        self.agents = nn.ModuleList()
        self._build_hierarchy()
        
        # Training params
        self.learning_rate = config.get('learning_rate', 1e-4)
        
        # How many agents to update per step (the ones most responsible for error)
        # Adaptive selection: agents above mean gradient get updated
        
        # Phase tracking
        self.current_iteration = 0
        self.phase_1_duration = config.get('phase_1_duration', 10000)
        self.phase_2_active = False
        
        self.to(self.device)
        
    def _build_hierarchy(self):
        """Build: Root → Managers → Specialists"""
        # Root agent
        root = Agent('root', 'Master', 'Root', 
                     self.embed_dim, self.hidden_dim, self.embed_dim, self.device)
        self.agents.append(root)
        
        # 4 Domain managers
        domains = ['Language', 'Logic', 'Pattern', 'Context']
        for domain in domains:
            manager = Agent(f'mgr_{domain}', domain, 'Manager',
                           self.embed_dim, self.hidden_dim, self.embed_dim, self.device)
            self.agents.append(manager)
            
            # 4 Specialists per manager
            subtopics = ['Basic', 'Advanced', 'Edge', 'Complex']
            for subtopic in subtopics:
                spec = Agent(f'spec_{domain}_{subtopic}', domain, subtopic,
                            self.embed_dim, self.hidden_dim, self.embed_dim, self.device)
                self.agents.append(spec)
        
        print(f"Built {len(self.agents)} agents: 1 root + 4 managers + 16 specialists")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - all agents contribute."""
        B, T = x.shape
        hidden = self.embedding(x)
        flat = hidden.reshape(B * T, self.embed_dim)
        
        # Each agent contributes equally
        agent_contributions = []
        for agent in self.agents:
            out = agent.forward(flat)
            agent_contributions.append(out)
        
        # Average all contributions
        combined = sum(agent_contributions) / len(self.agents)
        flat = flat + combined
        
        hidden = flat.reshape(B, T, self.embed_dim)
        logits = self.output_proj(hidden)
        
        return logits
    
    def find_responsible_agents(self) -> List[tuple]:
        """
        Find which agents are responsible for the current error.
        ADAPTIVE: Update agents with gradient above mean (not fixed top-K).
        """
        responsibilities = []
        
        for agent in self.agents:
            grad_sum = 0.0
            param_count = 0
            for param in agent.parameters():
                if param.grad is not None:
                    grad_sum += param.grad.abs().sum().item()
                    param_count += param.numel()
            
            if param_count > 0:
                responsibility = grad_sum / param_count
            else:
                responsibility = 0.0
            
            agent.error_responsibility = responsibility
            responsibilities.append((agent, responsibility))
        
        # ADAPTIVE SELECTION: Update agents above mean responsibility
        mean_resp = np.mean([r for _, r in responsibilities])
        
        # Select agents above mean (they're more responsible than average)
        responsible = [(a, r) for a, r in responsibilities if r > mean_resp]
        
        # Sort by responsibility (highest first)
        responsible.sort(key=lambda x: x[1], reverse=True)
        
        return responsible
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Sparse interpretable training:
        1. Forward pass
        2. Compute loss and gradients
        3. Find which agents caused the error
        4. Update ONLY those agents (not all!)
        """
        self.current_iteration += 1
        
        if self.current_iteration == self.phase_1_duration:
            print("\n" + "="*70)
            print("🚀 PHASE 2 ACTIVATED")
            print("="*70 + "\n")
            self.phase_2_active = True
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward + loss
        logits = self.forward(x)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        loss_val = loss.item()
        
        # Backprop to get gradients (but don't apply yet!)
        loss.backward()
        
        # ALWAYS update embedding and output_proj (shared infrastructure)
        with torch.no_grad():
            for param in self.embedding.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad
            for param in self.output_proj.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad
        
        # Find which agents are responsible (ADAPTIVE - above mean gradient)
        responsible_agents = self.find_responsible_agents()
        
        updated = 0
        with torch.no_grad():
            for agent, resp in responsible_agents:
                for param in agent.parameters():
                    if param.grad is not None:
                        param.data -= self.learning_rate * param.grad
                agent.total_updates += 1
                agent.total_errors_handled += resp
                updated += 1
        
        # Zero all gradients
        self.zero_grad()
        
        # Get names of updated agents for display
        updated_names = [a.id for a, _ in responsible_agents]
        
        return {
            'loss': loss_val,
            'updated': updated,
            'skipped': len(self.agents) - updated,
            'total_agents': len(self.agents),
            'avg_trust': 0.5,  # Placeholder for compatibility
            'high_trust': 0,
            'loo_computed': False,
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1',
            'responsible_agents': updated_names  # NEW: which agents learned
        }
    
    def get_stats(self) -> Dict:
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'total_agents': len(self.agents),
            'avg_trust': 0.5,
            'high_trust_agents': 0,
            'phase': 'Phase 2' if self.phase_2_active else 'Phase 1',
            'iteration': self.current_iteration
        }
    
    def get_agent_status(self) -> List[Dict]:
        """See which agents have learned the most."""
        return sorted([{
            'id': a.id,
            'domain': a.domain,
            'specialty': a.specialty,
            'trust': 0.5,
            'updated': a.total_updates,
            'skipped': self.current_iteration - a.total_updates,
            'loo_score': 0,
            'grad_score': round(a.error_responsibility, 6),
            'errors_handled': round(a.total_errors_handled, 4)
        } for a in self.agents], key=lambda x: x['updated'], reverse=True)
    
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
