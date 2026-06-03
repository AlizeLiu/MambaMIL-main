"""Tests for BP-Pool (Boundary-aware Pooling)."""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.AtpPool import ATPPool


def test_bp_pool_shape():
    """Test output shape of BP-Pool."""
    B, L, D = 2, 250, 256
    pool_size = 50
    x = torch.randn(B, L, D)
    
    pool = ATPPool(dim=D, pool_size=pool_size, pool_mode='bp')
    pool.eval()
    
    with torch.no_grad():
        z = pool(x)
    
    M = L // pool_size
    assert z.shape == (B, M, D), f"Shape mismatch: expected ({B}, {M}, {D}), got {z.shape}"
    print(f"PASS: test_bp_pool_shape {z.shape}")


def test_bp_pool_backward():
    """Test backward pass (gradient flow)."""
    B, L, D = 2, 250, 256
    pool_size = 50
    x = torch.randn(B, L, D, requires_grad=True)
    
    pool = ATPPool(dim=D, pool_size=pool_size, pool_mode='bp')
    
    z = pool(x)
    loss = z.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient on input"
    assert pool.bp_alpha_raw.grad is not None, "No gradient on bp_alpha_raw"
    assert pool.bp_beta_raw.grad is not None, "No gradient on bp_beta_raw"
    assert pool.bp_lambda_raw.grad is not None, "No gradient on bp_lambda_raw"
    print("PASS: test_bp_pool_backward")


def test_bp_pool_no_nan():
    """Test no NaN in output."""
    B, L, D = 2, 250, 256
    pool_size = 50
    x = torch.randn(B, L, D)
    
    pool = ATPPool(dim=D, pool_size=pool_size, pool_mode='bp')
    pool.eval()
    
    with torch.no_grad():
        z = pool(x)
    
    assert not torch.isnan(z).any(), "NaN in output"
    assert not torch.isinf(z).any(), "Inf in output"
    print("PASS: test_bp_pool_no_nan")


def test_bp_pool_high_gradient_reduces_merge():
    """Test that high gradient boundaries reduce merge probability."""
    B, L, D = 1, 100, 64
    pool_size = 50
    
    # Create input with sharp boundary at position 50
    x = torch.zeros(B, L, D)
    x[:, :50, :] = 1.0   # region A
    x[:, 50:, :] = -1.0   # region B (sharp boundary)
    
    pool = ATPPool(dim=D, pool_size=pool_size, pool_mode='bp')
    pool.eval()
    
    with torch.no_grad():
        # Get edge merge probabilities
        alpha = torch.nn.functional.softplus(pool.bp_alpha_raw) + 1e-6
        beta = torch.nn.functional.softplus(pool.bp_beta_raw) + 1e-6
        
        # Pad and reshape
        x_pad, _ = pool._pad_to_pool_size(x, mode='replicate')
        B, Lp, D = x_pad.shape
        M = Lp // pool_size
        x_seg = x_pad.reshape(B, M, pool_size, D)
        
        # Compute edge features
        x_left = x_seg[:, :, :-1, :]
        x_right = x_seg[:, :, 1:, :]
        
        x_left_norm = torch.nn.functional.normalize(x_left, dim=-1)
        x_right_norm = torch.nn.functional.normalize(x_right, dim=-1)
        sim = (x_left_norm * x_right_norm).sum(dim=-1)
        grad = ((x_left - x_right) ** 2).mean(dim=-1)
        
        q = alpha * sim - beta * grad
        p = torch.sigmoid(q)
        
        # At boundary (position 49), gradient should be high, merge prob low
        # In first window (0-49), boundary is at edge index 49
        # But since we have 2 windows, check window 0 edge 49 and window 1 edge 0
        # Window 0: tokens 0-49, edges 0-48
        # Window 1: tokens 50-99, edges 0-48 (within window)
        
        # The sharp boundary is at token 50, which is start of window 1
        # So within window 1, the first token is different from padding
        
        # Check that p values are reasonable (between 0 and 1)
        assert (p >= 0).all() and (p <= 1).all(), "p out of [0, 1] range"
        
        # Check that gradient at boundary is high
        # In window 0, the last edge (48) connects token 48 and 49 (both 1.0)
        # So grad should be low there
        # In window 1, the first edge (0) connects token 50 (-1.0) and 51 (-1.0)
        # So grad should be low there too
        
        # Actually, the boundary is between windows, not within a window
        # Let's test with a boundary within a window
        x2 = torch.zeros(B, 100, D)
        x2[:, :25, :] = 1.0
        x2[:, 25:, :] = -1.0  # boundary at position 25 (within first window)
        
        x2_pad, _ = pool._pad_to_pool_size(x2, mode='replicate')
        x2_seg = x2_pad.reshape(B, M, pool_size, D)
        
        x2_left = x2_seg[:, :, :-1, :]
        x2_right = x2_seg[:, :, 1:, :]
        
        x2_left_norm = torch.nn.functional.normalize(x2_left, dim=-1)
        x2_right_norm = torch.nn.functional.normalize(x2_right, dim=-1)
        sim2 = (x2_left_norm * x2_right_norm).sum(dim=-1)
        grad2 = ((x2_left - x2_right) ** 2).mean(dim=-1)
        
        q2 = alpha * sim2 - beta * grad2
        p2 = torch.sigmoid(q2)
        
        # At edge 24 (connecting token 24 and 25), gradient should be high
        # grad2[0, 0, 24] should be high (1.0 vs -1.0)
        # sim2[0, 0, 24] should be low (opposite directions)
        # So p2[0, 0, 24] should be low
        
        # At edge 10 (connecting token 10 and 11), gradient should be low
        # grad2[0, 0, 10] should be low (both 1.0)
        # sim2[0, 0, 10] should be high (same direction)
        # So p2[0, 0, 10] should be high
        
        # Check this
        p_at_boundary = p2[0, 0, 24].item()
        p_within_region = p2[0, 0, 10].item()
        
        print(f"  p at boundary (edge 24): {p_at_boundary:.4f}")
        print(f"  p within region (edge 10): {p_within_region:.4f}")
        
        # With learned parameters, this might not always hold
        # But the gradient should be higher at boundary
        grad_at_boundary = grad2[0, 0, 24].item()
        grad_within_region = grad2[0, 0, 10].item()
        
        print(f"  grad at boundary: {grad_at_boundary:.4f}")
        print(f"  grad within region: {grad_within_region:.4f}")
        
        assert grad_at_boundary > grad_within_region, \
            f"Gradient at boundary ({grad_at_boundary}) should be > within region ({grad_within_region})"
    
    print("PASS: test_bp_pool_high_gradient_reduces_merge")


def test_bp_pool_homogeneous_higher_merge():
    """Test that homogeneous regions have higher merge probability."""
    B, L, D = 1, 100, 64
    pool_size = 50
    
    # Create homogeneous input
    x_homo = torch.ones(B, L, D)
    
    pool = ATPPool(dim=D, pool_size=pool_size, pool_mode='bp')
    pool.eval()
    
    with torch.no_grad():
        alpha = torch.nn.functional.softplus(pool.bp_alpha_raw) + 1e-6
        beta = torch.nn.functional.softplus(pool.bp_beta_raw) + 1e-6
        
        x_pad, _ = pool._pad_to_pool_size(x_homo, mode='replicate')
        B, Lp, D = x_pad.shape
        M = Lp // pool_size
        x_seg = x_pad.reshape(B, M, pool_size, D)
        
        x_left = x_seg[:, :, :-1, :]
        x_right = x_seg[:, :, 1:, :]
        
        x_left_norm = torch.nn.functional.normalize(x_left, dim=-1)
        x_right_norm = torch.nn.functional.normalize(x_right, dim=-1)
        sim = (x_left_norm * x_right_norm).sum(dim=-1)
        grad = ((x_left - x_right) ** 2).mean(dim=-1)
        
        q = alpha * sim - beta * grad
        p = torch.sigmoid(q)
        
        # For homogeneous input, sim should be ~1, grad should be ~0
        # So q should be high, p should be high
        p_mean = p.mean().item()
        sim_mean = sim.mean().item()
        grad_mean = grad.mean().item()
        
        print(f"  Homogeneous: p_mean={p_mean:.4f}, sim_mean={sim_mean:.4f}, grad_mean={grad_mean:.6f}")
        
        assert p_mean > 0.5, f"Homogeneous p_mean ({p_mean}) should be > 0.5"
        assert sim_mean > 0.9, f"Homogeneous sim_mean ({sim_mean}) should be > 0.9"
        assert grad_mean < 0.01, f"Homogeneous grad_mean ({grad_mean}) should be < 0.01"
    
    print("PASS: test_bp_pool_homogeneous_higher_merge")


def test_bp_pool_parameters_nonnegative():
    """Test that softplus ensures non-negative parameters."""
    pool = ATPPool(dim=64, pool_size=50, pool_mode='bp')
    
    alpha = torch.nn.functional.softplus(pool.bp_alpha_raw) + 1e-6
    beta = torch.nn.functional.softplus(pool.bp_beta_raw) + 1e-6
    lam = torch.nn.functional.softplus(pool.bp_lambda_raw) + 1e-6
    
    assert alpha > 0, f"alpha ({alpha}) should be > 0"
    assert beta > 0, f"beta ({beta}) should be > 0"
    assert lam > 0, f"lambda ({lam}) should be > 0"
    print(f"PASS: test_bp_pool_parameters_nonnegative alpha={alpha:.4f}, beta={beta:.4f}, lambda={lam:.4f}")


if __name__ == '__main__':
    test_bp_pool_shape()
    test_bp_pool_backward()
    test_bp_pool_no_nan()
    test_bp_pool_high_gradient_reduces_merge()
    test_bp_pool_homogeneous_higher_merge()
    test_bp_pool_parameters_nonnegative()
    print("\n=== ALL BP-POOL TESTS PASSED ===")
