"""
Unit tests for ATPPool: avg, diffusion, residual modes.
Run: python tests/test_atp_pool.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from models.AtpPool import ATPPool

def test_shape():
    """Test output shapes for different input lengths."""
    print("=== Shape Test ===")
    
    # Exact divisible
    x = torch.randn(2, 2500, 256)
    for mode in ['avg', 'diffusion', 'residual']:
        pool = ATPPool(dim=256, pool_size=50, pool_mode=mode)
        y = pool(x)
        assert y.shape == (2, 50, 256), f"mode={mode}: expected (2,50,256), got {y.shape}"
        print(f"  {mode}: {x.shape} -> {y.shape} ✅")
    
    # Non-divisible
    x = torch.randn(2, 2513, 256)
    for mode in ['avg', 'diffusion', 'residual']:
        pool = ATPPool(dim=256, pool_size=50, pool_mode=mode)
        y = pool(x)
        assert y.shape == (2, 51, 256), f"mode={mode}: expected (2,51,256), got {y.shape}"
        print(f"  {mode}: {x.shape} -> {y.shape} ✅")
    
    print("  PASSED ✅\n")


def test_diffusion_steps_0_equals_avg():
    """diffusion_steps=0 must be equivalent to avg mode."""
    print("=== Diffusion Steps=0 == Avg Test ===")
    
    x = torch.randn(2, 2513, 256)
    
    pool_avg = ATPPool(dim=256, pool_size=50, pool_mode='avg')
    pool_diff0 = ATPPool(dim=256, pool_size=50, pool_mode='diffusion', diffusion_steps=0)
    
    # Copy weights to ensure same K
    pool_diff0.K.data = pool_avg.K.data.clone()
    
    y_avg = pool_avg(x)
    y_diff0 = pool_diff0(x)
    
    assert torch.allclose(y_avg, y_diff0, atol=1e-6), \
        f"Max diff: {(y_avg - y_diff0).abs().max().item()}"
    print(f"  Max diff: {(y_avg - y_diff0).abs().max().item():.2e} ✅")
    print("  PASSED ✅\n")


def test_residual_gamma_0_equals_avg():
    """residual mode with gamma_init=0 must be strictly equivalent to avg mode."""
    print("=== Residual gamma=0 == Avg Test (CRITICAL) ===")
    
    x = torch.randn(2, 2513, 256)
    
    pool_avg = ATPPool(dim=256, pool_size=50, pool_mode='avg')
    pool_res = ATPPool(
        dim=256, pool_size=50, pool_mode='residual',
        K_init=2.5, norm_type='mean',
        gamma_init=0.0, tau_init=2.0,
    )
    
    # Copy K to ensure same boundary computation
    pool_res.K.data = pool_avg.K.data.clone()
    
    y_avg = pool_avg(x)
    y_res = pool_res(x)
    
    max_diff = (y_avg - y_res).abs().max().item()
    assert torch.allclose(y_avg, y_res, atol=1e-6), \
        f"Max diff: {max_diff}"
    
    # Verify gamma is indeed 0
    gamma = torch.tanh(pool_res.gamma_raw).item()
    assert abs(gamma) < 1e-6, f"gamma should be 0, got {gamma}"
    
    print(f"  gamma = {gamma:.2e}")
    print(f"  Max diff: {max_diff:.2e} ✅")
    print("  PASSED ✅\n")


def test_backward():
    """Test backward pass for all modes."""
    print("=== Backward Test ===")
    
    for mode in ['avg', 'diffusion', 'residual']:
        x = torch.randn(2, 2513, 256, requires_grad=True)
        pool = ATPPool(dim=256, pool_size=50, pool_mode=mode, gamma_init=0.0)
        y = pool(x)
        loss = y.pow(2).mean()
        loss.backward()
        
        assert x.grad is not None, f"mode={mode}: x.grad is None"
        assert torch.isfinite(x.grad).all(), f"mode={mode}: x.grad has NaN/Inf"
        
        if mode == 'residual':
            assert pool.gamma_raw.grad is not None, "gamma_raw.grad is None"
            assert pool.tau_raw.grad is not None, "tau_raw.grad is None"
            print(f"  {mode}: x.grad ✅, gamma_raw.grad={pool.gamma_raw.grad.item():.6f} ✅")
        else:
            print(f"  {mode}: x.grad ✅")
    
    print("  PASSED ✅\n")


def test_no_nan():
    """No NaN for any mode."""
    print("=== No NaN Test ===")
    
    x = torch.randn(2, 2513, 256)
    
    for mode in ['avg', 'diffusion', 'residual']:
        pool = ATPPool(dim=256, pool_size=50, pool_mode=mode)
        y = pool(x)
        assert torch.isfinite(y).all(), f"mode={mode}: output has NaN/Inf"
        print(f"  {mode}: all finite ✅")
    
    print("  PASSED ✅\n")


def test_gpu_smoke():
    """GPU smoke test if available."""
    if not torch.cuda.is_available():
        print("=== GPU Smoke Test === SKIPPED (no CUDA)\n")
        return
    
    print("=== GPU Smoke Test ===")
    
    for mode in ['avg', 'diffusion', 'residual']:
        x = torch.randn(2, 2513, 256, device='cuda', requires_grad=True)
        pool = ATPPool(dim=256, pool_size=50, pool_mode=mode, gamma_init=0.0).cuda()
        y = pool(x)
        loss = y.pow(2).mean()
        loss.backward()
        
        assert torch.isfinite(y).all(), f"mode={mode}: GPU output has NaN/Inf"
        print(f"  {mode}: GPU forward+backward ✅")
    
    print("  PASSED ✅\n")


def test_diagnosis_residual():
    """Test diagnosis records for residual mode."""
    print("=== Diagnosis Test (residual) ===")
    
    pool = ATPPool(dim=256, pool_size=50, pool_mode='residual',
                   K_init=2.5, gamma_init=0.0)
    pool.enable_diagnosis(max_steps=3)
    
    for _ in range(5):
        x = torch.randn(1, 1000, 256)
        y = pool(x)
    
    summary = pool.get_diagnosis_summary()
    assert summary is not None, "No diagnosis records"
    assert 'gamma' in summary, "Missing gamma in diagnosis"
    assert 'K' in summary, "Missing K in diagnosis"
    assert 'boundary_mean' in summary, "Missing boundary_mean"
    
    # Should have exactly 3 records (max_steps=3)
    assert pool._diag_step == 3
    
    print(f"  Records: {pool._diag_step}")
    print(f"  gamma: {summary['gamma']['mean']:.6f}")
    print(f"  K: {summary['K']['mean']:.4f}")
    print(f"  boundary_mean: {summary['boundary_mean']['mean']:.6f}")
    print("  PASSED ✅\n")


if __name__ == "__main__":
    print("=" * 50)
    print("ATP-Pool Unit Tests")
    print("=" * 50)
    print()
    
    test_shape()
    test_diffusion_steps_0_equals_avg()
    test_residual_gamma_0_equals_avg()
    test_backward()
    test_no_nan()
    test_gpu_smoke()
    test_diagnosis_residual()
    
    print("=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)
