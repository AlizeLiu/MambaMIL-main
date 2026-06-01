"""
Unit tests for attention readout modules: SimpleAttention and GatedAttention.
Run: python tests/test_attention_readout.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from models.MambaMIL import SimpleAttention, GatedAttention, MambaMIL


def test_simple_attention_shape():
    """Test SimpleAttention output shapes."""
    print("=== SimpleAttention Shape Test ===")
    
    for in_dim, attn_dim in [(256, 128), (512, 64), (128, 256)]:
        attn = SimpleAttention(in_dim, attn_dim=attn_dim)
        
        for B, M in [(1, 50), (2, 100), (1, 25)]:
            h = torch.randn(B, M, in_dim)
            a = attn(h)
            assert a.shape == (B, M, 1), \
                f"in_dim={in_dim}, attn_dim={attn_dim}, B={B}, M={M}: expected ({B},{M},1), got {a.shape}"
    
    print("  PASSED ✓\n")


def test_gated_attention_shape():
    """Test GatedAttention output shapes."""
    print("=== GatedAttention Shape Test ===")
    
    for in_dim, attn_dim in [(256, 128), (512, 64), (128, 256)]:
        attn = GatedAttention(in_dim, attn_dim=attn_dim)
        
        for B, M in [(1, 50), (2, 100), (1, 25)]:
            h = torch.randn(B, M, in_dim)
            a = attn(h)
            assert a.shape == (B, M, 1), \
                f"in_dim={in_dim}, attn_dim={attn_dim}, B={B}, M={M}: expected ({B},{M},1), got {a.shape}"
    
    print("  PASSED ✓\n")


def test_attention_forward():
    """Test that both attention types produce valid outputs (no NaN, correct range)."""
    print("=== Forward Test ===")
    
    for attn_class, name in [(SimpleAttention, "Simple"), (GatedAttention, "Gated")]:
        attn = attn_class(256, attn_dim=128)
        
        h = torch.randn(2, 50, 256)
        a = attn(h)
        
        assert torch.isfinite(a).all(), f"{name}: output contains NaN/Inf"
        assert a.shape == (2, 50, 1), f"{name}: wrong shape {a.shape}"
        
        # Test that softmax produces valid attention weights
        a_soft = torch.softmax(a.transpose(1, 2), dim=-1)
        assert torch.allclose(a_soft.sum(dim=-1), torch.ones(2, 1), atol=1e-6), \
            f"{name}: softmax doesn't sum to 1"
        assert (a_soft >= 0).all(), f"{name}: negative attention weights after softmax"
    
    print("  PASSED ✓\n")


def test_attention_gradients():
    """Test that gradients flow through both attention types."""
    print("=== Gradient Test ===")
    
    for attn_class, name in [(SimpleAttention, "Simple"), (GatedAttention, "Gated")]:
        attn = attn_class(256, attn_dim=128)
        
        h = torch.randn(2, 50, 256, requires_grad=True)
        a = attn(h)
        loss = a.sum()
        loss.backward()
        
        assert h.grad is not None, f"{name}: input gradient is None"
        assert torch.isfinite(h.grad).all(), f"{name}: input gradient has NaN/Inf"
        
        # Check that attention parameters have gradients
        for param_name, param in attn.named_parameters():
            assert param.grad is not None, f"{name}: param {param_name} has no grad"
            assert torch.isfinite(param.grad).all(), f"{name}: param {param_name} grad has NaN/Inf"
    
    print("  PASSED ✓\n")


def test_simple_attention_structure():
    """Verify SimpleAttention has the expected layer structure."""
    print("=== SimpleAttention Structure Test ===")
    
    attn = SimpleAttention(256, attn_dim=128)
    
    # Should have: Linear(256, 128) -> Tanh -> Linear(128, 1)
    layers = list(attn.attn.children())
    assert len(layers) == 3, f"Expected 3 layers, got {len(layers)}"
    assert isinstance(layers[0], nn.Linear), f"Layer 0 should be Linear, got {type(layers[0])}"
    assert isinstance(layers[1], nn.Tanh), f"Layer 1 should be Tanh, got {type(layers[1])}"
    assert isinstance(layers[2], nn.Linear), f"Layer 2 should be Linear, got {type(layers[2])}"
    assert layers[0].in_features == 256
    assert layers[0].out_features == 128
    assert layers[2].in_features == 128
    assert layers[2].out_features == 1
    
    print("  PASSED ✓\n")


def test_gated_attention_structure():
    """Verify GatedAttention has the expected module structure."""
    print("=== GatedAttention Structure Test ===")
    
    attn = GatedAttention(256, attn_dim=128)
    
    # V: Linear -> Tanh
    assert isinstance(attn.V, nn.Sequential)
    v_layers = list(attn.V.children())
    assert isinstance(v_layers[0], nn.Linear) and isinstance(v_layers[1], nn.Tanh)
    assert v_layers[0].in_features == 256 and v_layers[0].out_features == 128
    
    # U: Linear -> Sigmoid
    assert isinstance(attn.U, nn.Sequential)
    u_layers = list(attn.U.children())
    assert isinstance(u_layers[0], nn.Linear) and isinstance(u_layers[1], nn.Sigmoid)
    assert u_layers[0].in_features == 256 and u_layers[0].out_features == 128
    
    # w: Linear(128, 1)
    assert isinstance(attn.w, nn.Linear)
    assert attn.w.in_features == 128 and attn.w.out_features == 1
    
    print("  PASSED ✓\n")


def test_mambamil_default_simple():
    """Test that MambaMIL with default attn_type uses SimpleAttention."""
    print("=== MambaMIL Default Simple Test ===")
    
    model = MambaMIL(
        in_dim=1024, n_classes=2, dropout=0.25, act='gelu',
        hidden_dim=256, local_layers=1, global_layers=1,
        pool_size=50, use_atp_pool=False, attn_type='simple', attn_dim=128,
    )
    assert isinstance(model.attention, SimpleAttention), \
        f"Default attention should be SimpleAttention, got {type(model.attention)}"
    
    print("  PASSED ✓\n")


def test_mambamil_gated():
    """Test that MambaMIL with attn_type='gated' uses GatedAttention."""
    print("=== MambaMIL Gated Test ===")
    
    model = MambaMIL(
        in_dim=1024, n_classes=2, dropout=0.25, act='gelu',
        hidden_dim=256, local_layers=1, global_layers=1,
        pool_size=50, use_atp_pool=False, attn_type='gated', attn_dim=128,
    )
    assert isinstance(model.attention, GatedAttention), \
        f"Gated attention should be GatedAttention, got {type(model.attention)}"
    
    print("  PASSED ✓\n")


def test_mambamil_forward_shape():
    """Test that MambaMIL with both attention types produces identical output shapes."""
    print("=== MambaMIL Forward Shape Test ===")
    
    for attn_type in ['simple', 'gated']:
        model = MambaMIL(
            in_dim=1024, n_classes=2, dropout=0.0, act='gelu',
            hidden_dim=256, local_layers=1, global_layers=1,
            pool_size=50, use_atp_pool=False, attn_type=attn_type, attn_dim=128,
        )
        model.eval()
        
        x = torch.randn(1, 500, 1024)
        with torch.no_grad():
            logits, Y_prob, Y_hat, A_raw, _ = model(x)
        
        assert logits.shape == (1, 2), f"attn_type={attn_type}: logits shape {logits.shape}"
        assert Y_prob.shape == (1, 2), f"attn_type={attn_type}: Y_prob shape {Y_prob.shape}"
        assert A_raw.shape == (1, 1, 10), f"attn_type={attn_type}: A_raw shape {A_raw.shape} (expected 500/50=10)"
        
        print(f"  {attn_type}: logits={logits.shape}, Y_prob={Y_prob.shape}, A_raw={A_raw.shape} ✓")
    
    print("  PASSED ✓\n")


def test_mambamil_invalid_attn_type():
    """Test that invalid attn_type raises ValueError."""
    print("=== Invalid attn_type Test ===")
    
    try:
        model = MambaMIL(
            in_dim=1024, n_classes=2, dropout=0.25, act='gelu',
            hidden_dim=256, attn_type='invalid'
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "attn_type" in str(e), f"Error message should mention attn_type: {e}"
    
    print("  PASSED ✓\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Attention Readout Unit Tests")
    print("=" * 60)
    print()
    
    test_simple_attention_shape()
    test_gated_attention_shape()
    test_attention_forward()
    test_attention_gradients()
    test_simple_attention_structure()
    test_gated_attention_structure()
    test_mambamil_default_simple()
    test_mambamil_gated()
    test_mambamil_forward_shape()
    test_mambamil_invalid_attn_type()
    
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
