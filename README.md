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

Run the example:
```bash
python k1_system/main.py
```

## Architecture

The system automatically transitions from Phase 1 (fixed parameters) to Phase 2 (self-learning) at iteration 10,000.

### Phase 2 Autonomous Adjustments

- Detects performance plateaus → Increases exploration
- Too many dead agents → Increases pruning
- High error complexity → Updates more agents
- Near capacity → Makes growth stricter

## Configuration

Edit `k1_system/config/config_phase1.json` to customize parameters.

## License

MIT License
