#!/usr/bin/env python3
"""
K-1 Self-Learning System - PyTorch Version with WikiText Dataset

This script trains the K-1 model using PyTorch for GPU acceleration
and WikiText-2 dataset for proper language modeling evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Device Setup
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# WikiText Dataset
# =============================================================================

def download_wikitext():
    """Download WikiText-2 dataset."""
    try:
        from datasets import load_dataset
        print("Loading WikiText-2 from HuggingFace...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        return dataset
    except ImportError:
        print("Installing datasets library...")
        os.system('pip install datasets')
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        return dataset


class WikiTextDataset(Dataset):
    """WikiText dataset for language modeling."""

    def __init__(self, texts: List[str], tokenizer, seq_len: int = 128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Concatenate all texts
        full_text = ' '.join([t for t in texts if t.strip()])

        # Tokenize
        self.tokens = tokenizer.encode(full_text)

        # Calculate number of sequences
        self.n_sequences = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start+1:end+1], dtype=torch.long)

        return x, y


class SimpleTokenizer:
    """Character-level tokenizer."""

    def __init__(self, texts: List[str]):
        # Build vocabulary from texts
        full_text = ' '.join([t for t in texts if t.strip()])
        chars = sorted(list(set(full_text)))

        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        print(f"Vocabulary size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.idx_to_char.get(t, '?') for t in tokens])


# =============================================================================
# K-1 Agent (PyTorch)
# =============================================================================

class Agent(nn.Module):
    """Single agent in the K-1 hierarchy."""

    def __init__(self, agent_id: str, input_dim: int, hidden_dim: int, output_dim: int,
                 max_children: int = 20, initial_trust: float = 0.3):
        super().__init__()

        self.agent_id = agent_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.routing = nn.Linear(hidden_dim, max_children)

        # Trust score (not a parameter, just tracking)
        self.register_buffer('trust', torch.tensor(initial_trust))
        self.register_buffer('activation_level', torch.tensor(0.0))

        # Hierarchy
        self.children: List['Agent'] = []
        self.parent: Optional['Agent'] = None

        # Cache
        self.last_hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through agent."""
        self.last_hidden = F.relu(self.fc1(x))
        output = self.fc2(self.last_hidden)
        self.activation_level = output.abs().mean().detach()
        return output

    def get_routing_scores(self) -> torch.Tensor:
        """Get routing scores for children."""
        if len(self.children) == 0 or self.last_hidden is None:
            return torch.tensor([])

        scores = self.routing(self.last_hidden)[:len(self.children)]
        return F.softmax(scores, dim=-1)

    def add_child(self, child: 'Agent'):
        """Add a child agent."""
        self.children.append(child)
        child.parent = self


# =============================================================================
# K-1 Model (PyTorch)
# =============================================================================

class K1ModelPyTorch(nn.Module):
    """K-1 Self-Learning Language Model in PyTorch."""

    def __init__(self, config: Dict):
        super().__init__()

        self.vocab_size = config['vocab_size']
        self.embed_dim = config.get('embed_dim', 256)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_agents = config.get('num_agents', 25)
        self.top_k = config.get('top_k', 5)
        self.phase1_steps = config.get('phase1_steps', 5000)

        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size)

        # Build agent hierarchy
        self.agents = nn.ModuleList()
        self.root_agent = self._build_hierarchy()

        # Training state
        self.current_step = 0
        self.phase_2_active = False

    def _build_hierarchy(self) -> Agent:
        """Build hierarchical agent structure."""
        # Root agent
        root = Agent('root', self.embed_dim, self.hidden_dim, self.embed_dim)
        self.agents.append(root)

        # Create 4 manager agents
        managers = []
        for i in range(4):
            mgr = Agent(f'mgr_{i}', self.embed_dim, self.hidden_dim, self.embed_dim)
            self.agents.append(mgr)
            root.add_child(mgr)
            managers.append(mgr)

        # Create leaf agents under each manager
        agent_idx = 0
        for mgr in managers:
            for j in range(5):
                agent = Agent(f'agent_{agent_idx}', self.embed_dim, self.hidden_dim, self.embed_dim)
                self.agents.append(agent)
                mgr.add_child(agent)
                agent_idx += 1

        return root

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input token indices (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq, embed)

        # Process each position through hierarchy
        outputs = []
        for t in range(seq_len):
            token_emb = embedded[:, t, :]  # (batch, embed)

            # Route through hierarchy (simplified - process through all agents)
            hidden = token_emb
            for agent in self.agents:
                agent_out = agent(hidden)
                hidden = hidden + 0.1 * agent_out  # Residual connection

            outputs.append(hidden)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq, embed)

        # Project to vocabulary
        logits = self.output_proj(output)  # (batch, seq, vocab)

        return logits

    def forward_with_routing(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """Forward with explicit routing through hierarchy."""
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)

        outputs = []
        all_activated = []

        for t in range(seq_len):
            token_emb = embedded[:, t, :]
            hidden, activated = self._route_through_hierarchy(token_emb)
            outputs.append(hidden)
            all_activated.append(activated)

        output = torch.stack(outputs, dim=1)
        logits = self.output_proj(output)

        return logits, all_activated

    def _route_through_hierarchy(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Agent]]:
        """Route input through hierarchy with exploration."""
        activated = []
        current = self.root_agent
        hidden = x

        while current is not None:
            # Forward through current agent
            out = current(hidden)
            hidden = hidden + 0.1 * out
            activated.append(current)

            # Check for children
            if len(current.children) == 0:
                break

            # Get routing scores and select next
            scores = current.get_routing_scores()
            if len(scores) == 0:
                break

            # Exploration vs exploitation
            if self.training and np.random.random() < 0.1:
                next_idx = np.random.randint(len(current.children))
            else:
                next_idx = scores.argmax().item()

            current = current.children[next_idx]

        return hidden, activated

    def get_current_phase(self) -> str:
        return 'Phase 2' if self.phase_2_active else 'Phase 1'

    def get_stats(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'num_agents': len(self.agents),
            'phase': self.get_current_phase(),
            'step': self.current_step
        }


# =============================================================================
# Baseline GPT (PyTorch)
# =============================================================================

class BaselineGPT(nn.Module):
    """Simple GPT-style model for comparison."""

    def __init__(self, config: Dict):
        super().__init__()

        self.vocab_size = config['vocab_size']
        self.embed_dim = config.get('embed_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.ff_dim = config.get('ff_dim', 1024)
        self.max_seq_len = config.get('max_seq_len', 128)
        self.dropout = config.get('dropout', 0.1)

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size)

        # Causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(self.max_seq_len))

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb

        # Transformer with causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(x, mask=mask)

        # Output projection
        logits = self.output_proj(x)

        return logits

    def get_stats(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, max_steps=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if max_steps and batch_idx >= max_steps:
            break

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        # Reshape for cross entropy
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device, max_batches=50):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = np.exp(min(avg_loss, 10))  # Cap to avoid overflow

    return avg_loss, perplexity


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100,
                  temperature: float = 0.8, device='cuda'):
    """Generate text from prompt."""
    model.eval()

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated = list(tokens[0].cpu().numpy())

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to max length
            input_tokens = tokens[:, -128:]

            logits = model(input_tokens)
            next_logits = logits[0, -1, :] / temperature

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("K-1 Self-Learning System vs Baseline GPT (PyTorch)")
    print("=" * 60)

    # Configuration
    config = {
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 4,
        'ff_dim': 1024,
        'max_seq_len': 128,
        'dropout': 0.1,
        'num_agents': 25,
        'top_k': 5,
        'phase1_steps': 2000,
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_epochs': 3,
        'max_steps_per_epoch': 500
    }

    # Load WikiText
    print("\n" + "-" * 40)
    print("Loading WikiText-2 Dataset")
    print("-" * 40)

    dataset = download_wikitext()

    # Create tokenizer
    all_texts = dataset['train']['text'] + dataset['validation']['text']
    tokenizer = SimpleTokenizer(all_texts)
    config['vocab_size'] = tokenizer.vocab_size

    # Create datasets
    train_dataset = WikiTextDataset(dataset['train']['text'], tokenizer, seq_len=config['max_seq_len'])
    val_dataset = WikiTextDataset(dataset['validation']['text'], tokenizer, seq_len=config['max_seq_len'])

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # Initialize models
    print("\n" + "-" * 40)
    print("Initializing Models")
    print("-" * 40)

    k1_model = K1ModelPyTorch(config).to(device)
    baseline_model = BaselineGPT(config).to(device)

    k1_stats = k1_model.get_stats()
    baseline_stats = baseline_model.get_stats()

    print(f"K-1 Model: {k1_stats['total_parameters']:,} parameters, {k1_stats['num_agents']} agents")
    print(f"Baseline GPT: {baseline_stats['total_parameters']:,} parameters")

    # Optimizers
    k1_optimizer = torch.optim.AdamW(k1_model.parameters(), lr=config['learning_rate'])
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=config['learning_rate'])

    # Training
    print("\n" + "=" * 60)
    print("Training K-1 Model")
    print("=" * 60)

    k1_history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        train_loss = train_epoch(k1_model, train_loader, k1_optimizer, device,
                                 max_steps=config['max_steps_per_epoch'])
        val_loss, val_ppl = evaluate(k1_model, val_loader, device)

        k1_history['train_loss'].append(train_loss)
        k1_history['val_loss'].append(val_loss)
        k1_history['val_ppl'].append(val_ppl)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    print("\n" + "=" * 60)
    print("Training Baseline GPT")
    print("=" * 60)

    baseline_history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        train_loss = train_epoch(baseline_model, train_loader, baseline_optimizer, device,
                                 max_steps=config['max_steps_per_epoch'])
        val_loss, val_ppl = evaluate(baseline_model, val_loader, device)

        baseline_history['train_loss'].append(train_loss)
        baseline_history['val_loss'].append(val_loss)
        baseline_history['val_ppl'].append(val_ppl)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    # Comparison
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    print("\n1. FINAL METRICS")
    print("-" * 40)
    print(f"  K-1 Final Val Loss:      {k1_history['val_loss'][-1]:.4f}")
    print(f"  Baseline Final Val Loss: {baseline_history['val_loss'][-1]:.4f}")
    print(f"  K-1 Final Perplexity:      {k1_history['val_ppl'][-1]:.2f}")
    print(f"  Baseline Final Perplexity: {baseline_history['val_ppl'][-1]:.2f}")

    print("\n2. PARAMETERS")
    print("-" * 40)
    print(f"  K-1 Parameters:      {k1_stats['total_parameters']:,}")
    print(f"  Baseline Parameters: {baseline_stats['total_parameters']:,}")

    # Generate samples
    print("\n3. GENERATION SAMPLES")
    print("-" * 40)

    prompt = "The quick brown fox"
    print(f"  Prompt: '{prompt}'")

    k1_sample = generate_text(k1_model, tokenizer, prompt, max_new_tokens=100, device=device)
    baseline_sample = generate_text(baseline_model, tokenizer, prompt, max_new_tokens=100, device=device)

    print(f"\n  K-1 Generated:\n    {k1_sample[:200]}...")
    print(f"\n  Baseline Generated:\n    {baseline_sample[:200]}...")

    # Winner
    print("\n4. OVERALL")
    print("-" * 40)
    if k1_history['val_ppl'][-1] < baseline_history['val_ppl'][-1]:
        print("  WINNER: K-1 Self-Learning Model!")
    else:
        print("  WINNER: Baseline GPT")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return k1_model, baseline_model, k1_history, baseline_history


if __name__ == "__main__":
    k1_model, baseline_model, k1_history, baseline_history = main()
