# Continual Learning Experiments

Test K-1's sparse updates vs baseline's catastrophic forgetting.

## 🎯 Experiment Overview

**Goal:** Test if K-1's 51% sparse updates prevent forgetting better than baseline's 100% updates.

**Setup:** Train both models on 3 datasets sequentially, then test text generation.

---

## 📋 Files Created

### Training Scripts (4 files):
1. **`train_k1_dataset1.py`** - Train K-1 on WikiText-2, save checkpoint
2. **`train_k1_dataset2.py`** - Continue K-1 on Dataset 2 (tests continual learning)
3. **`train_k1_dataset3.py`** - Final K-1 training on Dataset 3
4. **`train_baseline_all.py`** - Train baseline on all 3 datasets sequentially

### Generation Scripts (2 files):
5. **`generate_k1.py`** - Generate text with K-1 checkpoints
6. **`generate_baseline.py`** - Generate text with baseline checkpoints

---

## 🚀 How to Run

### Step 1: Train K-1 on 3 Datasets

```bash
# Dataset 1 (WikiText-2)
python3 train_k1_dataset1.py
# Saves: models/k1_dataset1.pt

# Dataset 2 (continues from dataset1)
python3 train_k1_dataset2.py
# Saves: models/k1_dataset2.pt

# Dataset 3 (continues from dataset2)
python3 train_k1_dataset3.py
# Saves: models/k1_final.pt
```

**Expected:** K-1's sparse updates should preserve old knowledge better.

---

### Step 2: Train Baseline on All Datasets

```bash
python3 train_baseline_all.py
# Trains on all 3 datasets sequentially
# Saves: models/baseline_dataset1.pt, baseline_dataset2.pt, baseline_final.pt
```

**Expected:** Baseline may "forget" Dataset 1 after learning Dataset 2/3 (catastrophic forgetting).

---

### Step 3: Test Text Generation

```bash
# Generate with K-1
python3 generate_k1.py

# Generate with Baseline
python3 generate_baseline.py
```

**Compare the outputs!** Does K-1 still generate coherent text after 3 datasets?

---

## 📊 What to Measure

### **Catastrophic Forgetting Test:**

After training on all 3 datasets:

1. **Load Dataset 1 checkpoint** (after first dataset only)
2. **Load Final checkpoint** (after all 3 datasets)
3. **Test on Dataset 1 prompts**

**If baseline forgot Dataset 1:**
- Final checkpoint performs worse on Dataset 1 prompts
- K-1 should perform better (sparse updates preserve knowledge)

### **Metrics:**

| Model | Dataset 1 Accuracy | Dataset 2 Accuracy | Dataset 3 Accuracy | Forgetting % |
|-------|-------------------|-------------------|-------------------|--------------|
| K-1 | ✓ | ✓ | ✓ | **Low** ✅ |
| Baseline | ❌ | ✓ | ✓ | **High** ⚠️ |

---

## 💡 Key Hypothesis

**K-1 (51% sparse updates)** should forget less than **Baseline (100% updates)** because:

1. Only 51% of parameters updated → Other 49% preserve old knowledge
2. Different parameter groups specialize in different domains
3. Sparse updates don't overwrite everything

**Baseline** will likely forget Dataset 1 because:
1. 100% parameter updates on Dataset 2/3 overwrite Dataset 1 knowledge
2. No preservation mechanism
3. Standard catastrophic forgetting

---

## 🔧 Advanced: Change Datasets

To test with different datasets, edit the training scripts:

```python
# In train_k1_dataset2.py, change line:
data_loader = DataLoader(dataset_name='wikitext', ...)

# To:
data_loader = DataLoader(dataset_name='openwebtext', ...)  # Different domain
# Or use code, math, scientific text, etc.
```

---

## 📁 Output Files

```
models/
├── k1_dataset1.pt          # K-1 after Dataset 1
├── k1_dataset2.pt          # K-1 after Dataset 2
├── k1_final.pt             # K-1 after all 3 datasets
├── baseline_dataset1.pt    # Baseline after Dataset 1
├── baseline_dataset2.pt    # Baseline after Dataset 2
└── baseline_final.pt       # Baseline after all 3 datasets
```

Use these to test at different stages!

---

## ✅ Success Criteria

K-1 demonstrates continual learning advantage if:

1. **Performance on Dataset 1 stays high** after training on 2 & 3
2. **Baseline performance on Dataset 1 drops** after training on 2 & 3
3. **Text generation quality remains good** across all domains
4. **Forgetting % lower** for K-1 than baseline

---

## 🎯 Quick Test

```bash
# 1. Train everything
python3 train_k1_dataset1.py
python3 train_k1_dataset2.py  
python3 train_k1_dataset3.py
python3 train_baseline_all.py

# 2. Generate and compare
python3 generate_k1.py > k1_output.txt
python3 generate_baseline.py > baseline_output.txt

# 3. Compare quality
diff k1_output.txt baseline_output.txt
```

**If K-1's output is more coherent → sparse updates win!** ✅
