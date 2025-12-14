# Google Colab Training Instructions

## Running the K-1 System on Google Colab

### Step 1: Open Google Colab
Go to [https://colab.research.google.com/](https://colab.research.google.com/)

### Step 2: Create a New Notebook
Click "New Notebook"

### Step 3: Clone and Run

Copy and paste this into a code cell:

```python
# Clone repository
!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1

# Run training
!python colab_run.py
```

### Step 4: Monitor Training

You'll see output like:

```
============================================================
Self-Learning K-1 System - WikiText-2 Training
============================================================

📦 Installing dependencies...

🔧 Importing K-1 System components...

============================================================
STARTING WIKITEXT-2 TRAINING
============================================================

📚 Loading WikiText-2 dataset...
Downloading WikiText-2 dataset...
Extracting dataset...
Download complete!
Loading WikiText-2 dataset...
Building vocabulary...
Converting text to sequences...
Dataset loaded:
  Training sequences: 201,764
  Validation sequences: 21,844
  Test sequences: 24,556
  Vocabulary size: 10,000

🚀 Initializing K-1 System...
Created language hierarchy with 13 agents

🎯 Starting training...
⏱️  Phase 1: Iterations 0-5,000 (Fixed Parameters)
⏱️  Phase 2: Iterations 5,000+ (Autonomous Adjustment)

Training will stop automatically when converged.

[Phase 1] Iter 0: Acc=0.0312, Loss=9.2103, Trust=0.300, Agents=13
[Phase 1] Iter 100: Acc=0.0625, Loss=8.8451, Trust=0.285, Agents=13
...
[Phase 1] Iter 4900: Acc=0.1250, Loss=7.5234, Trust=0.312, Agents=13

============================================================
🚀 PHASE 2 ACTIVATED at iteration 5000
Self-Learning Mode Enabled - Parameters Now Adjustable
============================================================

[Phase 2] Iter 5000: Acc=0.1406, Loss=7.3156, Trust=0.318, Agents=13
[Phase 2] Iter 5100: Acc=0.1562, Loss=7.1024, Trust=0.325, Agents=14

📊 Parameter Adjustments at Iteration 6000:
   • exploration_rate: +0.050 → 0.350
     Reason: long plateau - increase exploration

[Phase 2] Iter 10000: Acc=0.2188, Loss=6.2341, Trust=0.352, Agents=15
...

Stopping: plateau

============================================================
✅ TRAINING COMPLETE!
============================================================

📊 Check logs/ directory for detailed training logs
📈 Metrics saved to logs/metrics_*.json
```

### What Happens During Training

#### Phase 1 (Iterations 0-5,000)
- ✅ System trains with **fixed parameters**
- ✅ Trust scores develop naturally
- ✅ Builds baseline performance
- ✅ Creates trust cache of reliable agents

#### Phase 2 (Iterations 5,000+)
- 🚀 **Automatic activation** at iteration 5,000
- 🎯 **Parameter adjustments** based on performance
- 🔍 **Self-diagnosis** detects problems
- 🛠️ **Auto-correction** applies fixes
- 🏁 **Autonomous stopping** when converged

### Expected Training Time

On Google Colab (free tier):
- **Phase 1**: ~10-15 minutes
- **Phase 2**: ~15-30 minutes (depending on convergence)
- **Total**: ~25-45 minutes

### Viewing Results

After training completes, view the logs:

```python
# In a new cell:
!ls logs/

# View training log
!tail -50 logs/training_*.log

# View metrics
import json
import glob

metrics_file = glob.glob('logs/metrics_*.json')[0]
with open(metrics_file) as f:
    metrics = json.load(f)

print(f"Total iterations: {len(metrics)}")
print(f"Final accuracy: {metrics[-1]['accuracy']:.4f}")
print(f"Final loss: {metrics[-1]['loss']:.4f}")
print(f"Final agents: {metrics[-1]['total_agents']}")
```

### Customization

Edit the training parameters before running:

```python
# Before running colab_run.py, edit the file:
!nano colab_run.py

# Or modify in Python:
import sys
sys.path.insert(0, '.')

from colab_run import K1TextSystem, WikiText2Loader

# Custom configuration
data_loader = WikiText2Loader(vocab_size=5000)  # Smaller vocab
data_loader.load_data()

system = K1TextSystem(embedding_dim=64)  # Smaller embeddings
system.train(data_loader, num_iterations=10000)  # Shorter training
```

### Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in config
- Reduce `vocab_size` in data_loader
- Reduce `embedding_dim`

**Training too slow:**
- Reduce `max_iterations`
- Increase structural operation intervals

**Want to use GPU:**
```python
# Check GPU availability
!nvidia-smi

# The system will automatically use available compute
```

### Save Your Model

```python
# After training, save the hierarchy
import pickle

with open('trained_k1_model.pkl', 'wb') as f:
    pickle.dump(system.hierarchy, f)

# Download to your computer
from google.colab import files
files.download('trained_k1_model.pkl')
```

### Advanced: Live Monitoring

Monitor training in real-time:

```python
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Start training in background (requires modification)
# Then in another cell:
while True:
    clear_output(wait=True)
    
    # Read latest metrics
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Plot
    accuracies = [m['accuracy'] for m in metrics[-1000:]]
    plt.figure(figsize=(10, 4))
    plt.plot(accuracies)
    plt.title('Training Accuracy (Last 1000 iterations)')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()
    
    time.sleep(10)  # Update every 10 seconds
```

---

## Questions?

Open an issue on GitHub: https://github.com/PlanetDestroyyer/self-learning-k-1/issues
