import sys
import torch
sys.path.insert(0, '/home/x/projects/self-learning-k-1')
from models.k1_complete import create_k1_complete_model
from models.baseline_gpt_pytorch import BaselineGPTPyTorch

config = {'vocab_size': 100, 'embed_dim': 128, 'num_layers': 4, 'num_heads': 4, 'ff_dim': 512}

k1 = create_k1_complete_model(config)
baseline = BaselineGPTPyTorch(config)

x = torch.randint(0, 100, (4, 32))
y = torch.randint(0, 100, (4, 32))

# Test initial loss
with torch.no_grad():
    k1_logits = k1.forward(x.to(k1.device))
    k1_loss = torch.nn.functional.cross_entropy(k1_logits.reshape(-1, 100), y.to(k1.device).reshape(-1)).item()
    
    baseline_logits = baseline.forward(x.to(baseline.device))
    baseline_loss = torch.nn.functional.cross_entropy(baseline_logits.reshape(-1, 100), y.to(baseline.device).reshape(-1)).item()

print('STARTING LOSS COMPARISON:')
print(f'  K-1 starting loss:      {k1_loss:.2f}')
print(f'  Baseline starting loss: {baseline_loss:.2f}')
print(f'  Difference: {abs(k1_loss - baseline_loss):.2f}')

if abs(k1_loss - baseline_loss) < 1.0:
    print('OK - Same starting loss!')
else:
    print('DIFFERENT starting loss')
