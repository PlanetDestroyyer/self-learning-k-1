#!/usr/bin/env python3
"""
Test dataset loaders for WikiText, Code, and Scientific text.
Verifies that HuggingFace datasets can be loaded (or fallbacks work).
"""

import sys
import os
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.loader import DataLoader

def test_dataset(name, sample_len=500):
    print(f"\n{'='*50}")
    print(f"Testing Dataset: {name}")
    print(f"{'='*50}")
    
    try:
        loader = DataLoader(dataset_name=name, vocab_size=1000, seq_length=32)
        print(f"✓ Loader initialized")
        print(f"✓ Vocab size: {loader.get_vocab_size():,}")
        print(f"✓ Train data size: {len(loader.train_data):,}")
        
        # Show sample
        sample_indices = loader.train_data[0][0]
        sample_text = loader.decode(sample_indices)
        print(f"\nSample text ({sample_len} chars):")
        print(f"'{sample_text[:sample_len]}...'\n")
        
        return True
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Verifying Data Loaders...\n")
    
    # 1. WikiText
    if not test_dataset('wikitext'):
        print("Model training will fail without WikiText!")
    
    # 2. Code (Python)
    if not test_dataset('code_python'):
        print("Warning: Code dataset failed")
    
    # 3. Scientific
    if not test_dataset('scientific'):
        print("Warning: Scientific dataset failed")
        
    print("\nDataset Verification Complete")
