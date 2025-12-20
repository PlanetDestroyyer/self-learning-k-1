# K-1 Self-Learning System

**Hierarchical Path-Based Learning: Update Only What's Broken**

---

## 🧠 The Core Idea

**Traditional Backprop Problem:** Updates ALL weights every step, even those working perfectly. This causes:
- Catastrophic forgetting (old knowledge erased by new)
- Black box (no idea which parts learned what)
- Wasted compute (updating weights that don't need it)

**K-1 Solution:** Build a hierarchical tree of nodes. Only update the PATH responsible for errors.

```
                    ROOT (Manager)
                         |
         ┌───────────────┼───────────────┐
         |               |               |
      Node 1          Node 2          Node 3
         |               |               |
    ┌────┼────┐     ┌────┼────┐    ┌────┼────┐
   L1   L2   L3    L4   L5   L6   L7   L8   L9
```

### How It Works:

1. **Forward:** Data flows through tree
2. **Loss:** Computed at output  
3. **Backward:** Compute gradients for ALL nodes
4. **Analyze:** Which nodes have HIGH gradients? (causing errors)
5. **Update:** Only high-gradient nodes, skip the rest

```
Gradient Analysis:
  Root:    grad = 0.30  ✅ Update
  Node 1:  grad = 0.05  ❌ Skip (fine)
  Node 2:  grad = 0.45  ✅ Update (problem!)
  Node 3:  grad = 0.08  ❌ Skip
    └── L5: grad = 0.52  ✅ Update (culprit!)

Result: Update 3/13 nodes (23%)
        Preserve 77% of weights!
```

---

## 🚀 Quick Start

```bash
# Install
pip install torch datasets numpy

# Run K-1 experiment (3 datasets)
python experiment_k1.py

# Run baseline for comparison
python experiment_baseline.py

# Or just train K-1 on WikiText
python train_k1.py
```

---

## 📊 Experiments

### Continual Learning Test

Both experiments train on 3 datasets sequentially:
1. **WikiText-2** (general English)
2. **Code** (Python)
3. **Scientific** (research papers)

After each dataset, we evaluate on ALL previous datasets to measure **forgetting**.

| Script | Method | Expected Forgetting |
|--------|--------|-------------------|
| `experiment_k1.py` | K-1 (sparse path updates) | Low (~10-20%) |
| `experiment_baseline.py` | Traditional (update ALL) | High (~50%+) |

---

## 📁 Project Structure

```
self-learning-k-1/
├── train_k1.py              # Train K-1 on single dataset
├── experiment_k1.py         # K-1 continual learning experiment
├── experiment_baseline.py   # Baseline experiment for comparison
├── k1_system/
│   ├── core/
│   │   └── hierarchical_tree.py  # TreeNode + HierarchicalTree
│   └── config/
│       └── config_phase1.json
└── data/
    └── loader.py            # Dataset loading
```

---

## ⚙️ Configuration

```json
{
  "model": {
    "embed_dim": 128,
    "tree_depth": 3,          // Levels in tree
    "branching_factor": 3     // Children per node
  },
  "learning": {
    "top_k": 5,               // Update top 5 nodes
    "batch_size": 256
  }
}
```

---

## 🎯 Key Benefits

| Feature | Traditional | K-1 |
|---------|-------------|-----|
| **Params Updated** | 100% | ~25-40% |
| **Forgetting** | High | Low |
| **Explainability** | None | Full path tracking |
| **Debugging** | Hard | "Node 2 → L5 is broken" |

---

## 📄 License

MIT License
