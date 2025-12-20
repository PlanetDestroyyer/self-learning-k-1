# K-1 Self-Learning System

**Hierarchical Path-Based Learning: Update Only What's Broken**

---

## 🧠 The Core Idea

**Traditional Backprop Problem:** Updates ALL weights every step, even those working perfectly.

**K-1 Solution:** Build a hierarchical tree of nodes. Only update the PATH responsible for errors.

```
                    ROOT (Manager)
                         |
         ┌───────────────┼───────────────┐
         |               |               |
      Node 1          Node 2          Node 3
    (Manager)       (Manager)       (Manager)
         |               |               |
    ┌────┼────┐     ┌────┼────┐    ┌────┼────┐
    |    |    |     |    |    |    |    |    |
   L1   L2   L3    L4   L5   L6   L7   L8   L9
 (leaf)(leaf)(leaf)(leaf)(leaf)(leaf)(leaf)(leaf)(leaf)
```

### How It Works:

1. **Forward:** Data flows through the tree
2. **Loss:** Computed at output
3. **Backward:** Compute gradients for ALL nodes
4. **Analyze:** Which nodes have HIGH gradients? (responsible for errors)
5. **Update:** Only update high-gradient nodes, skip the rest

```
Example Gradient Analysis:
  Root:    grad = 0.30  ← Update
  Node 1:  grad = 0.05  ← Skip (working fine)
  Node 2:  grad = 0.45  ← Update (causing errors!)
  Node 3:  grad = 0.08  ← Skip
    └── L5:  grad = 0.52  ← Update (main culprit!)

Result: Update Root, Node 2, L5 only (3/13 = 23%)
        Skip 77% of weights - they're fine!
```

---

## 🎯 Why This Matters

| Problem | Traditional | K-1 Solution |
|---------|-------------|--------------|
| **Catastrophic Forgetting** | Updates all weights → old knowledge lost | Updates only error path → other knowledge preserved |
| **Explainability** | Black box - no idea which parts learned what | Clear path: "Node 2 → L5 learned this concept" |
| **Efficiency** | 100% params updated every step | ~25% params updated (those causing errors) |
| **Debugging** | Hard to find what's broken | Gradient tells you exactly which node is wrong |

---

## 🚀 Quick Start

```bash
# Install
pip install torch datasets numpy

# Train K-1 hierarchical system
python train_k1.py

# Train baseline for comparison
python train_baseline_all.py
```

---

## 📁 Project Structure

```
k1_system/
├── core/
│   └── hierarchical_tree.py    # TreeNode + HierarchicalTree + Trainer
├── config/
│   └── config_phase1.json      # Configuration
└── ...

train_k1.py                     # Main training script
data/loader.py                  # Data loading
```

---

## ⚙️ Configuration

```json
{
  "model": {
    "embed_dim": 128,
    "tree_depth": 3,          // Root + 2 levels of children
    "branching_factor": 3     // 3 children per node
  },
  "learning": {
    "top_k": 5,               // Update top 5 nodes per step
    "batch_size": 256
  }
}
```

**Tree Structure with depth=3, branching=3:**
- Level 0 (Root): 1 node
- Level 1 (Managers): 3 nodes
- Level 2 (Leaves): 9 nodes
- **Total: 13 nodes**
- **Updated per step: top_k=5 (38%)**

---

## 🔬 Research Contribution

### Novel Aspects:

1. **Hierarchical Tree for Language Modeling**
   - First application of path-based sparse updates to transformers

2. **Gradient-Based Path Selection**
   - Automatically identifies which part of the tree is "broken"

3. **Explainable AI**
   - "This sentence was processed by Path: Root → Node2 → Leaf5"
   - Know exactly which nodes learned which concepts

4. **Continual Learning**
   - New knowledge goes to high-gradient paths
   - Old knowledge stays in low-gradient paths (preserved)

---

## 📊 Expected Results

| Metric | Traditional | K-1 |
|--------|-------------|-----|
| Params Updated | 100% | ~25-40% |
| Speed | Baseline | Similar |
| Final Loss | X | ~X (similar) |
| Forgetting | High | Low (expected) |
| Interpretability | None | Full path tracking |

---

## 📄 License

MIT License
