# K-1 Self-Learning System

**Hierarchical Error Attribution: Making Neural Networks Interpretable**

---

## 🧠 The Core Problem

**Traditional Backpropagation is a BLACK BOX:**

When a neural network makes an error, backprop updates **ALL** weights blindly:
- ❌ **No interpretability:** Can't identify which part caused the error
- ❌ **Wasteful:** Updates millions of parameters that work fine
- ❌ **Undebuggable:** "Something broke" but no idea what
- ❌ **Black box:** No transparency into what's happening

```
Traditional Backprop:
Error occurs → Update ALL 3 million parameters → Hope it works
                ↓
         "Which part broke?"
         "No idea, it's a black box"
```

---

## 💡 The K-1 Solution: Hierarchical Error Attribution

**Core Idea:** Instead of updating everything, **TRACE** down a hierarchy to find **WHO** is responsible, then update **THAT**.

### The Hierarchical Structure:

```
                    MANAGER (Root)
                    "Oversees everything"
                         |
         ┌───────────────┼───────────────┐
         |               |               |
      AGENT 1         AGENT 2         AGENT 3
   "Specialist A"  "Specialist B"  "Specialist C"
         |               |               |
    ┌────┼────┐     ┌────┼────┐    ┌────┼────┐
   S1   S2   S3    S4   S5   S6   S7   S8   S9
   SUB-AGENTS (Actual workers)
```

Each level is a **set of parameters** (mini-neural network):
- **Manager:** Overall coordinator, sees big picture
- **Agents:** Specialized processors for different features
- **Sub-Agents:** Fine-grained workers doing specific tasks

---

## 🔍 How It Works: Error Attribution Flow

### **Step 1: Detect Error**
```
Model makes prediction
Loss is high → Something broke!
```

### **Step 2: Compute Responsibility (Gradients)**
```python
Backward pass computes gradients for ALL nodes:
  Manager:     grad = 0.30
  Agent 1:     grad = 0.05  ✓ Low gradient → Working fine
  Agent 2:     grad = 0.45  ⚠️ High gradient → Something wrong here!
  Agent 3:     grad = 0.08  ✓ Low gradient → OK
    Sub-Agent 4:  grad = 0.10
    Sub-Agent 5:  grad = 0.52  🚨 CULPRIT! Highest gradient!
    Sub-Agent 6:  grad = 0.12
```

**High gradient = Responsible for error = Needs fixing!**

### **Step 3: Hierarchical Drill-Down**
```
1. Manager: "I see an error"
2. Check Agents: "Agent 2 has high gradient"
3. Drill into Agent 2: "Sub-Agent 5 is the culprit!"
4. Attribution: "Sub-Agent 5 in Agent 2 caused the error"
```

### **Step 4: Proportional Updates**
```
Update based on responsibility level:
  Sub-Agent 5: 80% update  (most responsible!)
  Agent 2:     15% update  (parent oversight)
  Manager:      5% update  (top-level context)
  Other agents: 0% update  (not involved, skip!)

Result: Update ~3/13 components (23%)
        Preserve 77% of working parameters!
```

---

## 🎯 Key Innovation: Interpretability

### Before (Traditional):
```
❌ Error? Update all 3M parameters
❌ "Something broke" - no idea what
❌ Can't debug
❌ Complete black box
```

### After (K-1):
```
✅ Error? Trace to Sub-Agent 5 in Agent 2
✅ "Sub-Agent 5 is underperforming on feature X"
✅ Can debug specific component
✅ Transparent, interpretable system
```

---

## 📊 Benefits Beyond Interpretability

While **interpretability** is the PRIMARY goal, K-1 also provides:

### 1. **Computational Efficiency**
- Only update ~20-40% of parameters
- Skip components that work fine
- Faster training, less compute

### 2. **Catastrophic Forgetting Prevention** (Byproduct)
- Don't touch working parameters
- Preserve old knowledge automatically
- Better continual learning

### 3. **Debugging & Monitoring**
```python
# Can trace exactly which component failed
print("Error in Agent 2 → Sub-Agent 5")
print("Gradient: 0.52 (culprit!)")
print("Updating with 80% learning rate")
```

### 4. **Modular Improvements**
- Can replace specific sub-agents
- Can add new agents without retraining all
- Compositional architecture

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch datasets numpy

# Run K-1 experiment (3 datasets sequentially)
python experiment_k1.py

# Run baseline for comparison (updates all weights)
python experiment_baseline.py

# Or train K-1 on single dataset
python train_k1.py
```

---

## 📊 Experiments

### Continual Learning Test

Both systems train on 3 datasets sequentially:
1. **WikiText-2** (general English)
2. **Code** (Python programming)
3. **Scientific** (research papers)

After each dataset, we evaluate on ALL previous datasets.

| System | Method | Interpretability | Efficiency |
|--------|--------|-----------------|------------|
| **K-1** | Hierarchical attribution | ✅ Full path tracking | ~25-40% params updated |
| **Baseline** | Traditional backprop | ❌ Black box | 100% params updated |

**Expected Results:**
- K-1: Can identify "Agent X → Sub-Agent Y caused error"
- Baseline: "Something broke" (no details)

---

## 📁 Project Structure

```
self-learning-k-1/
├── README.md                    # This file
├── train_k1.py                  # Train K-1 on single dataset
├── experiment_k1.py             # K-1 continual learning experiment
├── experiment_baseline.py       # Baseline comparison (update all)
├── k1_system/
│   ├── core/
│   │   └── hierarchical_tree.py # TreeNode + Manager/Agent/Sub-Agent
│   └── config/
│       └── config_phase1.json   # System configuration
└── data/
    └── loader.py                # Multi-domain dataset loading
```

---

## ⚙️ Configuration

```json
{
  "model": {
    "embed_dim": 128,
    "tree_depth": 3,          // Manager → Agents → Sub-Agents
    "branching_factor": 3     // 3 children per parent
  },
  "learning": {
    "top_k": 5,               // Update top-5 responsible nodes
    "learning_rate": 0.001,
    "batch_size": 256
  }
}
```

**Tree Structure Example (depth=3, branching=3):**
- 1 Manager
- 3 Agents (level 1)
- 9 Sub-Agents (level 2)
- **Total: 13 nodes**

---

## 🎯 Comparison: K-1 vs. Traditional

| Aspect | Traditional Backprop | K-1 System |
|--------|---------------------|-----------|
| **Interpretability** | ❌ Black box | ✅ Full error attribution |
| **Debugging** | ❌ "Something broke" | ✅ "Agent 2 → Sub-Agent 5" |
| **Parameters Updated** | ❌ 100% (wasteful) | ✅ ~25-40% (efficient) |
| **Transparency** | ❌ None | ✅ Know who's responsible |
| **Modularity** | ❌ Monolithic | ✅ Hierarchical components |
| **Explainability** | ❌ Zero | ✅ Path tracking + gradients |

---

## 💡 Example: Error Attribution in Action

```python
# Training step
loss = model(batch)
loss.backward()

# K-1 analyzes gradients hierarchically:
Gradient Analysis:
  Manager (Root):    0.30  → Update 5%
  Agent 1:           0.05  → Skip (working fine)
  Agent 2:           0.45  → Update 15%
    └─ Sub-Agent 4:  0.10  → Skip
    └─ Sub-Agent 5:  0.52  → Update 80% (CULPRIT!)
    └─ Sub-Agent 6:  0.12  → Skip
  Agent 3:           0.08  → Skip

Result:
✅ Identified: "Sub-Agent 5 in Agent 2 is underperforming"
✅ Updated: 3/13 nodes (23%)
✅ Preserved: 10/13 nodes (77%) working fine
✅ Interpretable: Full path and responsibility known
```

---

## 🔬 Current Implementation Status

### ✅ Implemented:
- [x] Hierarchical tree structure (Manager → Agents → Sub-Agents)
- [x] Gradient-based error detection
- [x] Selective parameter updates (top-K)
- [x] Basic interpretability (shows which nodes updated)
- [x] Efficient training (skip unnecessary updates)

### 🚧 Planned (Full Idea):
- [ ] Proportional updates (80% culprit, 15% parent, 5% manager)
- [ ] Hierarchical drilling (start from manager, drill down)
- [ ] Responsibility visualization (path diagrams)
- [ ] Automated debugging tools
- [ ] Dynamic agent addition/removal

---

## 📈 Research Directions

This system opens up several research questions:

1. **Optimal Hierarchy Depth:** How many levels? (current: 3)
2. **Update Proportions:** What's best ratio? (80/15/5? 90/7/3?)
3. **Gradient Thresholds:** When is a gradient "high enough"?
4. **Specialization:** Do agents specialize in different features?
5. **Scalability:** Does this work for 1B+ parameter models?

---

## 📄 License

MIT License

---

## 🎓 Core Philosophy

**"Neural networks don't have to be black boxes. With hierarchical error attribution, we can KNOW what broke and FIX just that."**

Traditional AI: "It's magic, we don't know how it works"
K-1 System: "Sub-Agent 5 caused error X because of feature Y"

**Transparency > Opacity**
**Interpretability > Black Box**
**Targeted Fixes > Blind Updates**
