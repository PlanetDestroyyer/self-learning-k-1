#!/usr/bin/env python3
"""
Text Generation with K-1 Model (Modular Architecture)
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
from k1_system.core.modular_transformer import ModularSparseTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_k1_model(checkpoint_path='models/k1_final.pt'):
    """Load K-1 model from checkpoint"""
    print(f"Loading K-1 model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint.get('vocab_size', 10000)
    config = checkpoint.get('config', {})
    
    # Get config params or default
    embed_dim = config.get('model', {}).get('embed_dim', 128)
    ff_dim = config.get('model', {}).get('hidden_dim', 256)
    num_heads = config.get('model', {}).get('num_heads', 4)
    num_layers = config.get('model', {}).get('num_layers', 4)
    max_seq_len = config.get('model', {}).get('max_seq_len', 64)
    
    # Create model
    model = ModularSparseTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Load weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        print("Warning: Loading legacy checkpoint format not supported for modular architecture")
        return None, vocab_size, checkpoint
    
    print("✓ Model loaded!\n")
    return model, vocab_size, checkpoint


def generate_text(prompt, model, data_loader, max_tokens=50, temperature=0.8):
    """Generate text from prompt"""
    model.eval()
    
    # Tokenize prompt
    if hasattr(data_loader, 'tokenize'):
        tokens = data_loader.tokenize(prompt)
    else:
        # Fallback manual tokenization
        words = prompt.lower().split()
        tokens = [data_loader.word_to_idx.get(w, 0) for w in words]
        
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get last few tokens (context)
            context = generated[-64:] if len(generated) > 64 else generated
            x_tokens = torch.tensor(context, device=device) # [seq_len]
            
            # Forward pass - returns logits for all positions
            # logits shape: [seq_len, vocab_size]
            logits = model(x_tokens)
            
            # Get logits for the LAST token to predict the NEXT token
            next_token_logits = logits[-1, :]
            
            # Sample next token
            probs = torch.softmax(next_token_logits / temperature, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            
            # Stop at end token (if defined) or just continue
            # if next_token == data_loader.token_to_id.get('<eos>', -1):
            #    break
    
    # Decode
    text = data_loader.decode(generated)
    return text


if __name__ == '__main__':
    # Check for checkpoint
    model_path = 'models/k1_final.pt'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
    if not os.path.exists(model_path):
        # Try checking for other checkpoints
        for p in ['models/k1_dataset3.pt', 'models/k1_dataset2.pt', 'models/k1_dataset1.pt']:
            if os.path.exists(p):
                model_path = p
                break
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run training scripts first!")
        sys.exit(1)
    
    model, vocab_size, checkpoint = load_k1_model(model_path)
    
    # Load tokenizer WITH SAVED VOCABULARY from checkpoint
    if 'vocab' in checkpoint and 'word_to_idx' in checkpoint:
        print("Using vocabulary from checkpoint...")
        # Create loader just to get the decode function, then override vocab
        data_loader = DataLoader(dataset_name='wikitext', vocab_size=vocab_size, seq_length=64)
        data_loader.vocab = checkpoint['vocab']
        data_loader.word_to_idx = checkpoint['word_to_idx']
        data_loader.idx_to_word = checkpoint['idx_to_word']
        data_loader.token_to_id = data_loader.word_to_idx
    else:
        print("No vocab in checkpoint, using default WikiText vocab...")
        data_loader = DataLoader(dataset_name='wikitext', vocab_size=vocab_size, seq_length=64)
    
    print("="*70)
    print("K-1 TEXT GENERATION (Modular Architecture)")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Vocab size: {vocab_size:,}\n")
    
    # Test prompts
    prompts = [
        "The history of",
        "In the field of",
        "Scientists have discovered",
        "def main():",
        "The neural network model"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"PROMPT {i}: {prompt}")
        print('='*70)
        
        generated = generate_text(prompt, model, data_loader, max_tokens=50)
        print(generated)
    
    print(f"\n{'='*70}")
    print("Generation complete!")
