# K-1 Self-Learning System

**A Novel Approach to Neural Network Training**

Replace indiscriminate weight updates with **trust-based sparse learning** using hierarchical agents.

## 🎯 Core Innovation

```
Standard Backprop:     Error → Update ALL weights every step
K-1:                   Error → Find responsible agents → Update only those → Trust them not to repeat
```

| Aspect | Standard | K-1 |
|--------|----------|-----|
| Updates | 100% weights | ~20% (top-K agents) |
| Memory | None | Trust tracks "learned" |
| Interpretability | Black box | Know who handles what |

---

## 🧠 How It Works

### 1. Hierarchical Agents (Error Type Matching)
```
Root Manager
├── Language Manager    ← Language errors
│   ├── Basic          ← Grammar
│   ├── Advanced       ← Complex syntax
│   ├── Edge           ← Edge cases
│   └── Complex        ← Tricky patterns
├── Logic Manager       ← Logic errors
├── Pattern Manager     ← Pattern errors
└── Context Manager     ← Context errors

Total: 21 agents (1 root + 4 managers + 16 specialists)
```

### 2. Trust = "Already Learned" Cooldown
```python
# Trust states
LOW trust (0.0-0.3)  → Agent needs learning → UPDATE it
HIGH trust (0.7-1.0) → Agent already learned → SKIP it

# After update
agent.trust += 0.1  # "I trust you not to repeat this error"

# Slow decay (allows re-learning)
agent.trust *= 0.995  # High trust stays high, low fades to 0
```

### 3. Hybrid Attribution (Who Caused the Error?)

**Fast (every step):** Gradient magnitude
```python
contribution = sum(|gradient| for each parameter)
```

**Accurate (every 50 steps):** Leave-One-Out
```python
loss_without_agent = forward(exclude=agent)
loss_with_agent = forward()
contribution = loss_without - loss_with
# Positive = agent was HELPING (removing hurts)
# Negative = agent was HURTING (removing helps)
```

---

## 🚀 Quick Start

### Google Colab
```python
!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1
!pip install torch numpy matplotlib datasets
!python run_colab.py
```

### Local
```bash
git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
cd self-learning-k-1
pip install -r requirements.txt
python run_colab.py
```

---

## 📊 Training Output

```
Step   10: Loss=4.82 [LOO COMPUTED]
  root:         trust=0.77, loo=+0.0005 (helping)
  mgr_Language: trust=0.00, loo=-0.0001 (hurting)
  spec_Basic:   trust=0.20, loo=+0.0001 (helping)

Step  100: Loss=3.21 | Updated=5 | Skipped=8 | Trust=0.45
Step  200: Loss=2.87 | Updated=5 | Skipped=12 | Trust=0.58
...

🚀 PHASE 2 ACTIVATED: Self-Learning Mode Enabled

Step 10100: Loss=1.95 | Updated=5 | Skipped=16 | Trust=0.78
```

---

## ⚙️ Configuration

```python
config = {
    'vocab_size': 1000,
    'embed_dim': 128,
    'hidden_dim': 256,
    'top_k': 5,              # Update top-5 agents per step
    'trust_threshold': 0.7,  # Skip agents with trust > 0.7
    'trust_increase': 0.1,   # Trust boost after update
    'trust_decay': 0.995,    # Slow trust decay
    'loo_interval': 50,      # Compute LOO every 50 steps
}
```

---

## 📁 Project Structure

```
self-learning-k-1/
├── models/
│   ├── k1_complete.py         # ⭐ Main K-1 implementation
│   └── baseline_gpt_pytorch.py # Baseline for comparison
├── run_colab.py               # ⭐ Training script
└── config.json                # Configuration
```

---

## 🔬 Research Value

### Novel Contributions
1. **Trust as Cooldown** - Agents promise not to repeat errors
2. **Sparse Updates** - Only update responsible agents
3. **Hybrid Attribution** - Gradient (fast) + LOO (accurate)
4. **Hierarchical Error Matching** - Domains handle error types

### Potential Applications
- Interpretable AI (know which component learned what)
- Continual learning (sparse updates = less forgetting)
- Modular networks (add/remove agents)

### Honest Limitations
- Still uses backprop (for gradient computation)
- May train slower than full backprop
- Hyperparameter sensitive

---

## 📚 API

```python
from models.k1_complete import create_k1_complete_model

model = create_k1_complete_model(config)

# Train
metrics = model.train_step(x, y)
# Returns: loss, updated, skipped, avg_trust, loo_computed

# Get agent status
status = model.get_agent_status()
# Returns: id, domain, trust, loo_score, grad_score

# Generate
tokens = model.generate(prompt, max_new_tokens=50)
```

---

## 📄 License

MIT

## 📖 Citation

```bibtex
@misc{k1-self-learning,
  title={K-1: Trust-Based Sparse Learning with Hierarchical Agents},
  year={2024},
  url={https://github.com/PlanetDestroyyer/self-learning-k-1}
}
```
