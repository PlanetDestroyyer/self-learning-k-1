# Self-Learning K-1 System

**K-1 = Knowledge Hierarchical System with Self-Learning Capability**

A hierarchical neural network that combines human-designed initialization with autonomous self-learning and structural evolution.

## Overview

The K-1 System is a novel neural network architecture that:

1. **Hierarchical Structure**: Organizes knowledge in a multi-level tree (Manager → Agent → Sub-agent)
2. **Human-Designed Initialization**: Leverages domain knowledge for initial structure
3. **Self-Learning Evolution**: Autonomously optimizes structure and parameters

## Key Features

### Two-Phase Operation

- **Phase 1 (Iterations 0-10,000)**: Fixed parameters, establish baseline
- **Phase 2 (Iterations 10,000+)**: Autonomous parameter adjustment and full self-learning

### Core Components

- **Trust System**: Assigns trust scores (0.0-1.0) to agents based on performance
- **Hierarchical Routing**: Routes inputs through the tree to find specialist agents
- **Credit Assignment**: Trust-based responsibility assignment instead of backpropagation
- **Structural Operations**:
  - **Self-Pruning**: Removes unused/low-performing agents
  - **Self-Merging**: Combines redundant agents
  - **Self-Growing**: Creates new specialists for knowledge gaps
  - **Self-Reorganization**: Optimizes hierarchy structure

### Autonomy Features

- **Parameter Controller**: Automatically adjusts hyperparameters based on performance
- **Stopping Controller**: Decides when to stop training
- **Self-Diagnostic**: Detects and corrects problems
- **Rollback System**: Reverts harmful structural changes

## Quick Start

### 🔥 Train 50M Parameter Model (BEST - VERIFIABLE LEARNING)

Train a proper **50 million parameter** language model with comprehensive evaluation:

**Google Colab:**
```python
!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1
!pip install -r requirements.txt
!python train_10m_model.py
```

**Local:**
```bash
git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
cd self-learning-k-1
pip install -r requirements.txt
python train_10m_model.py
```

**After training, evaluate:**
```bash
python evaluate_model.py
```

**What you get:**
- ✅ **52M parameters** (96 hierarchical agents)
- ✅ **Real dataset** (WikiText-2 - auto-downloaded)
- ✅ **Perplexity tracking** (proper language modeling metric)
- ✅ **Baseline comparison** (vs random model)
- ✅ **Learning verification** (proves it's actually learning!)
- ✅ **Visualization** (training curves saved as PNG)
- ✅ **Agent specialization analysis**
- ✅ **Saved model** (`trained_k1_50m.pkl`)

### Train on WikiText-2 (Smaller Scale)

**Google Colab (Easiest):**
```python
# In a Colab notebook cell:
!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1
!python colab_run.py
```

**Local Machine:**
```bash
# Clone repository
git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
cd self-learning-k-1

# Install dependencies
pip install -r requirements.txt

# Run training on WikiText-2
python colab_run.py
```

The system will:
1. 📚 Download WikiText-2 dataset automatically
2. 🏗️ Build language-specific hierarchy (Syntax, Semantics, Vocabulary)
3. 🎯 Train with automatic Phase 1 → Phase 2 transition
4. 📊 Save logs and metrics to `logs/` directory

### Simple Demo (Synthetic Data)

```python
from k1_system.main import K1System
import numpy as np

# Create dataset
train_data = np.random.randn(1000, 128)
train_labels = np.random.randint(0, 10, 1000)

# Initialize and train
system = K1System()
system.train(train_data, train_labels)
```

## Architecture

### 50M Parameter Model Architecture

**Parameter Breakdown:**
```
Embeddings:    11.52M  (30,000 vocab × 384 dim)
Agents (96):   28.91M  (96 agents × ~301k params each)
Output Proj:   11.52M  (384 dim × 30,000 vocab)
────────────────────────
TOTAL:         51.95M parameters
```

**Hierarchy Structure (96 agents):**
```
Language Model (root)
├─ Syntax Manager (18 agents)
├─ Semantics Manager (18 agents)
├─ Vocabulary Manager (18 agents)
├─ Context Manager (18 agents)
└─ Structure Manager (18 agents)
```

**Per-Agent Architecture:**
- Input: 384-dim
- Hidden: 384-dim (ReLU activation)
- Output: 384-dim
- Routing: 384 → 10 (for child selection)

The system automatically transitions from Phase 1 (fixed parameters 0-10,000 iterations) to Phase 2 (autonomous self-learning 10,000+ iterations).

### WikiText-2 Training Setup

**Language-Specific Hierarchy:**
```
Master Manager (Language)
├─ Syntax Manager
│  ├─ Grammar Agent
│  ├─ Punctuation Agent
│  └─ Structure Agent
├─ Semantics Manager
│  ├─ Meaning Agent
│  ├─ Context Agent
│  └─ Relations Agent
└─ Vocabulary Manager
   ├─ CommonWords Agent
   ├─ RareWords Agent
   └─ Entities Agent
```

**Training Details:**
- **Dataset**: WikiText-2 (2M tokens, ~30K vocabulary)
- **Task**: Next-word prediction (language modeling)
- **Phase 1**: Iterations 0-5,000 (baseline)
- **Phase 2**: Iterations 5,000+ (autonomous optimization)
- **Embeddings**: 128-dimensional word vectors
- **Batch Size**: 32 sequences of 50 tokens

### Phase 2 Autonomous Adjustments

- Detects performance plateaus → Increases exploration
- Too many dead agents → Increases pruning
- High error complexity → Updates more agents
- Near capacity → Makes growth stricter

## Configuration

Edit `k1_system/config/config_phase1.json` to customize parameters.

## License

MIT License
