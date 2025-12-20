#!/bin/bash
# Quick setup script for Google Colab
# Run with: bash setup_colab.sh

echo "=================================================="
echo "K-1 SYSTEM - GOOGLE COLAB SETUP"
echo "=================================================="

# Check if running on Colab
if [ -d "/content" ]; then
    echo "✓ Detected Google Colab environment"
    cd /content
else
    echo "⚠️  Not running on Colab, using current directory"
fi

# Clone or update repository
if [ -d "self-learning-k-1" ]; then
    echo "✓ Repository exists, pulling latest changes..."
    cd self-learning-k-1
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/PlanetDestroyyer/self-learning-k-1.git
    cd self-learning-k-1
fi

echo ""
echo "Installing dependencies..."
pip install -q torch datasets numpy

echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=================================================="
echo "SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Quick test (5 min):  python3 test_soft_routing.py"
echo "  2. Full training (30 min): python3 compare_baseline_vs_k1.py"
echo ""
echo "See COLAB_SETUP.md for detailed instructions"
echo "=================================================="
