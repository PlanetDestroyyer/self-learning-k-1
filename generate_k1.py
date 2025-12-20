#!/usr/bin/env python3
"""
Text Generation with K-1 Model
Loads saved checkpoint and generates text
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_k1_model(checkpoint_path='models/k1_final.pt'):
    """Load K-1 model from checkpoint"""
    print(f"Loading K-1 model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint['vocab_size']
    
    # Recreate model architecture
    embed_dim = 128
    hidden_dim = 256
    output_dim = 128
    
    embedding = nn.Embedding(vocab_size, embed_dim).to(device)
    network = nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    ).to(device)
    output_proj = nn.Linear(output_dim, vocab_size).to(device)
    
    # Load weights
    embedding.load_state_dict(checkpoint['embedding'])
    network.load_state_dict(checkpoint['network'])
    output_proj.load_state_dict(checkpoint['output_proj'])
    
    print("✓ Model loaded!\n")
    return embedding, network, output_proj, vocab_size


def generate_text(prompt, embedding, network, output_proj, data_loader, max_tokens=50, temperature=0.8):
    """Generate text from prompt"""
    embedding.eval()
    network.eval()
    output_proj.eval()
    
    # Tokenize prompt
    tokens = data_loader.tokenize(prompt)
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get last few tokens (context)
            context = generated[-64:] if len(generated) > 64 else generated
            x_tokens = torch.tensor(context, device=device)
            
            # Forward pass
            x_emb = embedding(x_tokens)
            x_pool = torch.mean(x_emb, dim=0)
            hidden = network(x_pool)
            logits = output_proj(hidden)
            
            # Sample next token
            probs = torch.softmax(logits / temperature, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            
            # Stop at end token
            if next_token == data_loader.token_to_id.get('<eos>', -1):
                break
    
    # Decode
    text = data_loader.decode(generated)
    return text


if __name__ == '__main__':
    # Load model (change path to test different checkpoints)
    model_path = 'models/k1_final.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run train_k1_dataset1.py first!")
        sys.exit(1)
    
    embedding, network, output_proj, vocab_size = load_k1_model(model_path)
    
    # Load data loader for tokenization
    data_loader = DataLoader(dataset_name='wikitext', vocab_size=10000, seq_length=64)
    
    print("="*70)
    print("K-1 TEXT GENERATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Vocab size: {vocab_size:,}\n")
    
    # Test prompts
    prompts = [
        "The history of",
        "In the field of",
        "Scientists have discovered"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"PROMPT {i}: {prompt}")
        print('='*70)
        
        generated = generate_text(prompt, embedding, network, output_proj, data_loader, max_tokens=30)
        print(generated)
    
    print(f"\n{'='*70}")
    print("Generation complete!")
    print("Try different checkpoints (k1_dataset1.pt, k1_dataset2.pt, k1_final.pt)")
    print("to test continual learning!")
