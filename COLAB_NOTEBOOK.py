"""
K-1 SYSTEM - GOOGLE COLAB QUICK TEST
Copy and paste this entire cell into Google Colab
"""

# ==========================================
# STEP 1: SETUP (Run once)
# ==========================================
print("🔧 Setting up K-1 System on Google Colab...")

!git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
%cd self-learning-k-1
!pip install -q torch datasets numpy

# Verify GPU
import torch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# STEP 2: RUN QUICK TEST (5 minutes)
# ==========================================
print("\n" + "="*70)
print("🧪 TESTING SOFT ROUTING FIX (1000 steps, ~5 minutes)")
print("="*70)
print("This tests if the soft routing fix works:")
print("  - Expected: Update % should jump from 4.3% to 30-50%")
print("  - If successful, K-1 will learn much better!")
print("="*70 + "\n")

!python3 test_soft_routing.py

# ==========================================
# STEP 3: CHECK RESULTS
# ==========================================
import json

try:
    with open('test_soft_routing_results.json') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("📊 FINAL VERDICT")
    print("="*70)
    print(f"Update percentage: {results['update_percentage']:.1f}%")
    print(f"Success: {'YES ✅' if results['success'] else 'NO ❌'}")
    print("="*70)
    
    if results['success']:
        print("\n🎉 SOFT ROUTING WORKS!")
        print("\nNext steps:")
        print("  1. Run full training: !python3 compare_baseline_vs_k1.py")
        print("  2. Test continual learning advantage")
        print("  3. Build interpretability demos")
    else:
        print("\n⚠️  Soft routing needs more tuning")
        print("Check the output above for details")
        
except FileNotFoundError:
    print("\n❌ Test didn't complete - check output above for errors")

# ==========================================
# OPTIONAL: RUN FULL TRAINING (~30 min)
# ==========================================
# Uncomment the lines below to run full comparison:
#
# print("\n🚀 Running full baseline vs K-1 comparison...")
# !python3 compare_baseline_vs_k1.py
