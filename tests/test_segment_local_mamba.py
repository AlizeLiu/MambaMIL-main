"""
Tests for segment-wise Local Mamba.
Run: python tests/test_segment_local_mamba.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch


def test_shape_no_local():
    """Shape test with local_layers=0, global_layers=0."""
    print("=== Shape Test (no local/global layers) ===")
    from models.MambaMIL import MambaMIL
    model = MambaMIL(
        in_dim=1024, n_classes=4, dropout=0.0, act='gelu',
        survival=True, layer=0, rate=5, type='SRMamba',
        hidden_dim=256, local_layers=0, global_layers=0,
        pool_size=50, use_atp_pool=True, diffusion_steps=0,
        K_init=2.5, atp_dt=0.1, norm_type='mean', pool_mode='avg',
        local_segment_mode='chunk', local_segment_size=50,
    )
    x = torch.randn(1, 2500, 1024)
    with torch.no_grad():
        out = model(x)
    hazards = out[0]
    assert hazards.shape == (1, 4), f"Expected (1, 4), got {hazards.shape}"
    print(f"  Input: {x.shape} -> hazards: {hazards.shape} ✅")
    print("  PASSED ✅\n")


def test_segment_vs_flat_no_local():
    """With local_layers=0, segment-wise and flat should produce same pooled output."""
    print("=== Segment vs Flat (no local layers) ===")
    from models.MambaMIL import MambaMIL

    torch.manual_seed(42)
    model_flat = MambaMIL(
        in_dim=1024, n_classes=4, dropout=0.0, act='gelu',
        survival=True, layer=0, rate=5, type='SRMamba',
        hidden_dim=256, local_layers=0, global_layers=0,
        pool_size=50, use_atp_pool=True, diffusion_steps=0,
        K_init=2.5, atp_dt=0.1, norm_type='mean', pool_mode='avg',
        local_segment_mode='none', local_segment_size=50,
    )

    torch.manual_seed(42)
    model_seg = MambaMIL(
        in_dim=1024, n_classes=4, dropout=0.0, act='gelu',
        survival=True, layer=0, rate=5, type='SRMamba',
        hidden_dim=256, local_layers=0, global_layers=0,
        pool_size=50, use_atp_pool=True, diffusion_steps=0,
        K_init=2.5, atp_dt=0.1, norm_type='mean', pool_mode='avg',
        local_segment_mode='chunk', local_segment_size=50,
    )

    # Copy weights
    model_seg.load_state_dict(model_flat.state_dict())

    x = torch.randn(1, 2500, 1024)
    with torch.no_grad():
        out_flat = model_flat(x)
        out_seg = model_seg(x)

    # Compare hazards (first element of tuple)
    assert torch.allclose(out_flat[0], out_seg[0], atol=1e-5), \
        f"Flat and segment should match with no local layers. Max diff: {(out_flat[0] - out_seg[0]).abs().max()}"
    print(f"  Max diff: {(out_flat[0] - out_seg[0]).abs().max():.2e} ✅")
    print("  PASSED ✅\n")


def test_non_divisible_length():
    """Non-divisible input length should not crash (pad logic)."""
    print("=== Non-Divisible Length ===")
    from models.MambaMIL import MambaMIL
    model = MambaMIL(
        in_dim=1024, n_classes=4, dropout=0.0, act='gelu',
        survival=True, layer=0, rate=5, type='SRMamba',
        hidden_dim=256, local_layers=0, global_layers=0,
        pool_size=50, use_atp_pool=True, diffusion_steps=0,
        K_init=2.5, atp_dt=0.1, norm_type='mean', pool_mode='avg',
        local_segment_mode='chunk', local_segment_size=50,
    )
    x = torch.randn(1, 2530, 1024)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out[0]).all(), "Output has NaN/Inf"
    print(f"  Input: {x.shape} -> hazards: {out[0].shape} ✅")
    print("  PASSED ✅\n")


def test_brpool_compatibility():
    """BRPool + segment-wise should work (forward + backward)."""
    print("=== BRPool Compatibility ===")
    from models.MambaMIL import MambaMIL
    model = MambaMIL(
        in_dim=1024, n_classes=4, dropout=0.0, act='gelu',
        survival=True, layer=0, rate=5, type='SRMamba',
        hidden_dim=256, local_layers=0, global_layers=0,
        pool_size=50, use_atp_pool=True, diffusion_steps=0,
        K_init=2.5, atp_dt=0.1, norm_type='mean',
        pool_mode='residual', gamma_init=0.0, tau_init=2.0,
        local_segment_mode='chunk', local_segment_size=50,
    )
    x = torch.randn(1, 2500, 1024)
    out = model(x)
    loss = out[0].pow(2).mean()
    loss.backward()
    assert torch.isfinite(out[0]).all(), "Output has NaN/Inf"
    print(f"  Forward + backward ✅")
    print(f"  gamma_raw.grad = {model.atp_pool.gamma_raw.grad.item():.6f}")
    print("  PASSED ✅\n")


def test_pool_size_gt_segment_size():
    """pool_size > local_segment_size should raise ValueError."""
    print("=== Pool Size > Segment Size ===")
    from models.MambaMIL import MambaMIL
    try:
        model = MambaMIL(
            in_dim=1024, n_classes=4, dropout=0.0, act='gelu',
            survival=True, layer=0, rate=5, type='SRMamba',
            hidden_dim=256, local_layers=0, global_layers=0,
            pool_size=100, use_atp_pool=True, diffusion_steps=0,
            K_init=2.5, atp_dt=0.1, norm_type='mean', pool_mode='avg',
            local_segment_mode='chunk', local_segment_size=50,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Raised ValueError: {e} ✅")
    print("  PASSED ✅\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Segment-wise Local Mamba Tests")
    print("=" * 50)
    print()
    test_shape_no_local()
    test_segment_vs_flat_no_local()
    test_non_divisible_length()
    test_brpool_compatibility()
    test_pool_size_gt_segment_size()
    print("=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)
