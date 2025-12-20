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

## 🧪 Test 1: Quick Soft Routing Test (5 minutes)

```python
!python3 test_soft_routing.py
```

**What this tests:**
- Whether soft routing activates more agents (5-10 instead of 2-3)
- Expected: Update % should jump from 4.3% to 30-50%

**Expected output:**
```
✅ SUCCESS: Soft routing is working!
   Update % jumped from 4.3% to 35.2%
```

---

## 🧪 Test 2: Full Comparison (30 minutes)

If Test 1 succeeds, run the full baseline vs K-1 comparison:

```python
!python3 compare_baseline_vs_k1.py
```

**What this does:**
- Trains baseline for 1 epoch (57,708 steps)
- Trains K-1 for 1 epoch with soft routing
- Compares performance side-by-side

**Expected improvements:**
- K-1 perplexity: 7,576 → ~800-1,200 (10x better!)
- K-1 update %: 4.3% → 30-50%

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
!python3 test_soft_routing.py 2>&1 | tee test_output.log
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
| 5 min | Soft routing test | Update % = 30-50% ✅ |
| 30 min | Full comparison | Perplexity < 1,200 ✅ |
| 2 hours | Continual learning | K-1 forgets <20% vs baseline 50%+ ✅ |

---

## 🎯 Success Criteria

### Minimum Viable Success
- ✅ Soft routing increases update % to >30%
- ✅ K-1 perplexity drops below 2,000
- ✅ All agents participate (no starvation)

### Stretch Goals
- ✅ K-1 perplexity approaches baseline (~400-600)
- ✅ Demonstrates continual learning advantage
- ✅ Clear agent specializations identified

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
!python3 test_soft_routing.py

# 4. Check Results
import json
with open('test_soft_routing_results.json') as f:
    results = json.load(f)
    
print(f"\n📊 Update percentage: {results['update_percentage']:.1f}%")
if results['success']:
    print("✅ Soft routing working! Ready for full training.")
    
    # Ask user if they want to continue
    run_full = input("\nRun full training? (30 min) [y/n]: ")
    
    if run_full.lower() == 'y':
        print("\n🚀 Starting full training...")
        !python3 compare_baseline_vs_k1.py
else:
    print("⚠️  Soft routing needs tuning. Review test output above.")
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
