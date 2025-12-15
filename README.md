# K-1 Self-Learning System

**K-1 = Knowledge Hierarchical System with Trust-Based Self-Learning**

A novel neural network architecture that combines hierarchical agent organization with trust-based credit assignment and structural plasticity for autonomous learning.

## Overview

The K-1 System is an experimental approach to neural network learning that:

1. **Hierarchical Agents**: Organizes computation into a tree of specialized agents
2. **Trust-Based Credit Assignment**: Uses trust scores to determine which agents to update (instead of updating all weights)
3. **Hybrid Learning**: Trust selects WHICH agents, gradients determine HOW to update them
4. **Structural Plasticity**: Autonomously prunes, merges, grows, and reorganizes agents

## Key Innovation

Traditional backpropagation updates ALL weights based on error gradients. K-1 takes a different approach:

```
Traditional Backprop:     Error → Update ALL weights
K-1 Hybrid Approach:      Error → Trust selects TOP-K agents → Gradients update those agents
```

This creates **sparse, targeted updates** that may be more efficient for certain types of learning.

## Project Structure

```
self-learning-k-1/
├── core/                    # Core components
│   ├── agent.py            # Neural network agents
│   ├── hierarchy.py        # Hierarchical organization
│   ├── routing.py          # Input routing with exploration
│   └── trust.py            # Trust score management
├── learning/                # Learning algorithms
│   ├── forward.py          # Forward pass computation
│   ├── backward.py         # Proper gradient backpropagation
│   └── credit.py           # Trust-based credit assignment
├── structural/              # Structural plasticity
│   ├── pruning.py          # Remove low-trust agents
│   ├── merging.py          # Combine similar agents
│   ├── growing.py          # Create new specialists
│   └── reorganization.py   # Optimize hierarchy
├── autonomy/                # Autonomous control
│   ├── parameter_controller.py  # Auto-adjust hyperparameters
│   ├── stopping.py         # Early stopping logic
│   └── diagnostic.py       # Self-diagnosis
├── models/                  # Model implementations
│   ├── k1_model.py         # K-1 Self-Learning Language Model
│   └── baseline_gpt.py     # Baseline GPT for comparison
├── data/                    # Data loading
│   └── loader.py           # Dataset utilities
├── utils/                   # Utilities
│   ├── metrics.py          # Evaluation metrics
│   └── logger.py           # Logging
├── safety/                  # Safety mechanisms
│   ├── snapshot.py         # Model checkpointing
│   └── validation.py       # Change validation
├── config.json              # Configuration file
├── run_colab.py            # Main training script (K-1 vs Baseline comparison)
└── requirements.txt         # Dependencies
```

## Quick Start

### Google Colab (Recommended)

```python
# Clone repository
!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1

# Install dependencies
!pip install numpy matplotlib

# Run comparison experiment
!python run_colab.py
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
cd self-learning-k-1

# Install dependencies
pip install -r requirements.txt

# Run training and comparison
python run_colab.py
```

## What the Experiment Does

`run_colab.py` trains TWO models and compares them:

1. **K-1 Self-Learning Model**:
   - Hierarchical agent structure
   - Trust-based credit assignment
   - Sparse updates (only top-K agents per step)
   - Two-phase learning (fixed → autonomous)
   - Structural plasticity in Phase 2

2. **Baseline GPT Model**:
   - Standard transformer architecture
   - Full backpropagation
   - Fixed architecture

The script compares:
- Training loss
- Perplexity
- Parameter efficiency
- Generated text quality

## Two-Phase Learning

### Phase 1: Fixed Architecture (First 50% of training)
- Learn basic patterns
- Build agent specializations
- Establish trust scores
- NO structural changes

### Phase 2: Autonomous Learning (Last 50% of training)
- Enable structural plasticity
- Autonomous hyperparameter adjustment
- Self-pruning of low-trust agents
- Self-merging of similar agents
- Self-growing for knowledge gaps

## Architecture Details

### K-1 Model Components

**Agent Structure**:
```
Agent {
  weights: {W1, b1, W2, b2}     # Two-layer MLP
  trust_score: float            # 0.0 - 1.0
  responsibility: float         # How much this agent contributed
  children: List[Agent]         # Child agents in hierarchy
}
```

**Trust-Based Credit Assignment**:
```python
# Select top-K agents by trust-weighted responsibility
scores = responsibility * (1 + trust_score)
top_k_agents = agents.sorted_by(scores)[:K]

# Update only selected agents using gradients
for agent in top_k_agents:
    gradient = compute_gradient(agent, error)
    agent.weights -= learning_rate * gradient
    agent.trust += improvement  # Actual measured improvement
```

**Structural Operations**:
- **Pruning**: Remove agents with trust < threshold
- **Merging**: Combine agents with similarity > threshold
- **Growing**: Add agents when existing ones are overloaded
- **Reorganization**: Move agents to better parents

### Baseline GPT Model

Standard transformer with:
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Positional embeddings

## Configuration

Edit `config.json` to customize:

```json
{
  "model": {
    "vocab_size": 256,
    "embed_dim": 128,
    "hidden_dim": 256
  },
  "k1_system": {
    "hierarchy": {
      "depth": 3,
      "branching_factor": 4
    },
    "routing": {
      "top_k": 4,
      "exploration_rate": 0.1
    },
    "trust": {
      "initial_trust": 0.5
    }
  },
  "training": {
    "learning_rate": 0.0001,
    "max_steps": 10000
  }
}
```

## Research Context

### What K-1 Explores

1. **Sparse Updates**: Can we achieve similar learning by updating only relevant weights?
2. **Trust as Credit**: Can trust scores provide useful learning signals beyond gradients?
3. **Structural Plasticity**: Can networks benefit from dynamic architecture changes?
4. **Hybrid Approaches**: How to combine traditional gradients with alternative credit assignment?

### Current Limitations

- Pure trust-based credit (without gradients) doesn't provide enough learning signal
- Structural operations need careful validation to avoid catastrophic changes
- Exploration-exploitation tradeoff in routing is challenging
- May require more training steps than standard backprop

### Potential Applications

- Interpretable AI (agents have clear responsibilities)
- Continual learning (add new agents without forgetting)
- Resource-efficient inference (route to relevant agents only)
- Modular neural networks

## API Reference

### K1SelfLearningLM

```python
from models.k1_model import K1SelfLearningLM

# Initialize
model = K1SelfLearningLM({
    'vocab_size': 256,
    'embed_dim': 128,
    'hidden_dim': 256,
    'hierarchy_depth': 3,
    'branching_factor': 4,
    'learning_rate': 0.0001
})

# Train step
loss = model.train_step(input_ids, target_ids)

# Generate
tokens = model.generate(prompt_ids, max_new_tokens=50)

# Get statistics
stats = model.get_stats()
```

### BaselineGPT

```python
from models.baseline_gpt import BaselineGPT

# Initialize
model = BaselineGPT({
    'vocab_size': 256,
    'embed_dim': 128,
    'num_layers': 4,
    'num_heads': 4,
    'learning_rate': 0.0001
})

# Train step
loss = model.train_step(input_ids, target_ids)

# Generate
tokens = model.generate(prompt_ids, max_new_tokens=50)
```

## Results Interpretation

When running `run_colab.py`, you'll see:

1. **Loss Comparison**: Lower is better
2. **Perplexity**: Lower means better language modeling
3. **Parameter Efficiency**: Loss per million parameters
4. **Generation Samples**: Qualitative text comparison

The K-1 system is experimental - it may or may not outperform the baseline depending on:
- Dataset characteristics
- Training duration
- Hyperparameter tuning
- Random initialization

## Contributing

This is a research project exploring alternative neural network learning approaches. Contributions welcome!

## License

MIT License

## Citation

If you use this code in your research:

```
@misc{k1-self-learning,
  title={K-1: A Hierarchical Self-Learning System with Trust-Based Credit Assignment},
  year={2024},
  url={https://github.com/PlanetDestroyyer/self-learning-k-1}
}
```
