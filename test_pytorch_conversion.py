#!/usr/bin/env python3
"""
Quick sanity tests for PyTorch conversion.

This script tests the core components after converting from NumPy to PyTorch.
"""

import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/x/projects/self-learning-k-1')

from k1_system.core.agent import Agent


def test_agent_forward():
    """Test Agent forward pass works with PyTorch."""
    print("Testing Agent forward pass...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # Create agent
    agent = Agent(
        agent_id="test_agent",
        input_dim=128,
        hidden_dim=256,
        output_dim=128
    ).to(device)

    print(f"  Agent created: {agent.id}")
    print(f"  Agent device: {agent.device}")

    # Test with tensor input
    x_tensor = torch.randn(32, 128).to(device)
    output_tensor = agent.forward(x_tensor)

    assert output_tensor.shape == (32, 128), f"Wrong shape: {output_tensor.shape}"
    assert output_tensor.device == device, f"Wrong device: {output_tensor.device}"
    assert isinstance(output_tensor, torch.Tensor), "Output should be tensor"
    print(f"  ✓ Tensor input: shape {output_tensor.shape}, device {output_tensor.device}")

    # Test with numpy input (should auto-convert)
    x_numpy = np.random.randn(32, 128).astype(np.float32)
    output_numpy = agent.forward(x_numpy)

    assert output_numpy.shape == (32, 128), f"Wrong shape: {output_numpy.shape}"
    assert isinstance(output_numpy, torch.Tensor), "Output should be tensor"
    print(f"  ✓ NumPy input: auto-converted, shape {output_numpy.shape}")

    # Test single sample (1D input)
    x_single = torch.randn(128).to(device)
    output_single = agent.forward(x_single)

    assert output_single.shape == (128,), f"Wrong shape: {output_single.shape}"
    print(f"  ✓ Single sample: shape {output_single.shape}")

    print("✅ Agent forward pass works!\n")


def test_agent_routing():
    """Test Agent routing works with PyTorch."""
    print("Testing Agent routing...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create parent and children
    parent = Agent(agent_id="parent", output_dim=128).to(device)
    child1 = Agent(agent_id="child1").to(device)
    child2 = Agent(agent_id="child2").to(device)

    parent.add_child(child1)
    parent.add_child(child2)

    print(f"  Parent has {len(parent.child_agents)} child agents")

    # Test routing
    x = torch.randn(128).to(device)
    routing_probs = parent.route(x)

    assert routing_probs.shape == (2,), f"Wrong shape: {routing_probs.shape}"
    assert torch.abs(routing_probs.sum() - 1.0) < 1e-5, "Routing probs should sum to 1"
    assert (routing_probs >= 0).all(), "Routing probs should be >= 0"
    print(f"  ✓ Routing probs: {routing_probs.detach().cpu().numpy()} (sum={routing_probs.sum():.4f})")

    # Test routing parameter expansion
    child3 = Agent(agent_id="child3").to(device)
    parent.add_child(child3)

    assert parent.routing.shape[1] >= 3, f"Routing should expand, got shape {parent.routing.shape}"
    print(f"  ✓ Routing parameter expanded to {parent.routing.shape}")

    print("✅ Agent routing works!\n")


def test_autograd():
    """Test gradient computation with PyTorch autograd."""
    print("Testing PyTorch autograd...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create agent
    agent = Agent(input_dim=128, hidden_dim=256, output_dim=128).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    # Generate data
    x = torch.randn(32, 128).to(device)
    target = torch.randn(32, 128).to(device)

    # Forward pass
    output = agent.forward(x)
    loss = torch.mean((output - target) ** 2)

    print(f"  Initial loss: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist for layer parameters (routing not used in forward)
    grad_count = 0
    for name, param in agent.named_parameters():
        if 'layer' in name:  # Only check layer1 and layer2
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                print(f"  ✓ {name}: grad_norm={grad_norm:.6f}")
            else:
                print(f"  ✗ {name}: NO GRADIENT")
                assert False, f"No gradient for {name}"
        elif param.grad is None:
            print(f"  - {name}: not used in forward() (expected)")

    print(f"  ✓ All {grad_count} layer parameters have gradients")

    # Optimizer step
    optimizer.step()

    # Forward pass again
    output2 = agent.forward(x)
    loss2 = torch.mean((output2 - target) ** 2)

    print(f"  Loss after 1 step: {loss2.item():.6f}")
    assert loss2.item() < loss.item(), "Loss should decrease after optimization"
    print(f"  ✓ Loss decreased by {(loss.item() - loss2.item()):.6f}")

    print("✅ Autograd works!\n")


def test_device_handling():
    """Test proper device handling."""
    print("Testing device handling...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    agent = Agent(input_dim=64, hidden_dim=128, output_dim=64).to(device)

    # Check all parameters are on correct device
    for name, param in agent.named_parameters():
        assert param.device == device, f"{name} on wrong device: {param.device}"

    print(f"  ✓ All parameters on {device}")

    # Test input device mismatch handling
    if device.type == 'cuda':
        x_cpu = torch.randn(16, 64)  # CPU tensor
        output = agent.forward(x_cpu)  # Should auto-move to GPU
        assert output.device.type == 'cuda', "Output should be on CUDA"
        print(f"  ✓ Auto-moved CPU input to GPU")

    print("✅ Device handling works!\n")


def test_parameter_count():
    """Test parameter count matches expected."""
    print("Testing parameter count...")

    agent = Agent(input_dim=128, hidden_dim=256, output_dim=128)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Expected:
    # layer1: (128 * 256) + 256 = 33,024
    # layer2: (256 * 128) + 128 = 32,896
    # routing: 128 * 10 = 1,280
    # Total: 67,200

    expected = (128 * 256 + 256) + (256 * 128 + 128) + (128 * 10)
    assert total_params == expected, f"Expected {expected:,}, got {total_params:,}"

    print(f"  ✓ Parameter count correct: {total_params:,}")
    print("✅ Parameter count matches!\n")


if __name__ == '__main__':
    print("="*70)
    print("PyTorch Conversion Test Suite")
    print("="*70)
    print()

    try:
        test_agent_forward()
        test_agent_routing()
        test_autograd()
        test_device_handling()
        test_parameter_count()

        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)

    except Exception as e:
        print()
        print("="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
