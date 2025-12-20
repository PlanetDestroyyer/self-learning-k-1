# K-1 System - Google Colab Setup & Testing

## 🚀 Quick Start on Google Colab

### Step 1: Clone Repository
```python
!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1
```

### Step 2: Install Dependencies
```python
!pip install torch datasets numpy
```

### Step 3: Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🧪 Test 1: Quick Architecture Test (5 minutes)

```python
!python3 test_modular_architecture.py
```

**What this tests:**
- Modular transformer architecture with sparse updates
- Data loading and vocabulary building
- Training loop with proper autoregressive loss

**Expected output:**
```
✓ If you see this, the new architecture works!
✓ Modular Transformer with proper autoregressive loss is functional!
```

---

## 🧪 Test 2: Full Training (30 minutes)

If Test 1 succeeds, run full training on datasets:

```python
# Train K-1 on WikiText-2
!python3 train_k1_dataset1.py

# Train baseline for comparison
!python3 train_baseline_all.py
```

**What this does:**
- Trains K-1 for 1 epoch (57,708 steps) with sparse updates
- Trains baseline with traditional backpropagation
- Saves model checkpoints

**Expected results:**
- K-1 update %: ~50% (sparse updates working)
- Both models should show decreasing perplexity

---

## 🧪 Test 3: Continual Learning (Advanced)

If K-1 works well, test continual learning:

```python
# Coming soon: test_continual_learning.py
```

---

## 📊 Monitoring Progress

### Check Current Status
```python
# Look at validation logs
!tail -20 compare_baseline_vs_k1.log
```

### Watch Training in Real-Time
```python
# Stream output
!python3 test_modular_architecture.py 2>&1 | tee test_output.log
```

---

## ⚠️ Troubleshooting

### Import Errors
```python
import sys
sys.path.insert(0, '/content/self-learning-k-1')
```

### CUDA Out of Memory
```python
# Reduce batch size in config
!sed -i 's/"batch_size": 64/"batch_size": 32/' k1_system/config/config_phase1.json
```

### Data Loading Issues
```python
# Clear huggingface cache and retry
!rm -rf ~/.cache/huggingface/datasets
```

---

## 📈 Expected Results Timeline

| Time | Test | Expected Result |
|------|------|-----------------|
| 5 min | Architecture test | Update % = ~50%, Training completes ✅ |
| 30 min | Full training | Models train successfully ✅ |
| 2 hours | Continual learning | K-1 handles multiple datasets ✅ |

---

## 🎯 Success Criteria

### Minimum Viable Success
- ✅ Sparse updates working (~50% parameter updates)
- ✅ Data loading and vocabulary building works
- ✅ Training completes without errors

### Stretch Goals
- ✅ Multiple dataset training succeeds
- ✅ Model checkpoints save correctly
- ✅ Generation scripts produce coherent text

---

## 📝 Full Colab Notebook Template

Copy this to a new Colab cell:

```python
# ==========================================
# K-1 SYSTEM - COMPLETE COLAB SETUP
# ==========================================

# 1. Setup
print("🔧 Setting up K-1 System...")
!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1
!pip install -q torch datasets numpy

# 2. Verify GPU
import torch
print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# 3. Quick Test (5 min)
print("\n" + "="*70)
print("🧪 RUNNING QUICK TEST (1000 steps, ~5 minutes)")
print("="*70)
!python3 test_modular_architecture.py

# 4. Full Training (optional)
print("\n✓ Quick test complete!")
run_full = input("\nRun full training on WikiText-2? (30 min) [y/n]: ")

if run_full.lower() == 'y':
    print("\n🚀 Starting K-1 training...")
    !python3 train_k1_dataset1.py

    print("\n🚀 Starting baseline training...")
    !python3 train_baseline_all.py

    print("\n✓ Training complete! Check models/ directory for checkpoints.")
```

---

## 🔗 Quick Links

- **GitHub:** https://github.com/PlanetDestroyyer/self-learning-k-1
- **Issues:** Report bugs or ask questions there
- **Docs:** See `/home/x/.gemini/antigravity/brain/*/` for detailed analysis

---

## ⏱️ Time Estimates (on T4 GPU)

- Quick test: **5 minutes** (1,000 steps)
- Full comparison: **30 minutes** (57,708 steps)  
- Continual learning: **2 hours** (3 domains × 5K steps)
