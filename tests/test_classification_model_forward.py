"""
Test MambaMIL classification forward pass with IHG-Mamba parameters.
All tests use synthetic data on CUDA (SRMamba requires CUDA).
"""
import sys
import torch

sys.path.insert(0, '.')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Helpers
# ============================================================

def _make_model(**overrides):
    """Create a MambaMIL with IHG defaults, allowing override of any param."""
    from models.MambaMIL import MambaMIL
    defaults = dict(
        in_dim=1024,
        n_classes=2,
        dropout=0.0,
        act='gelu',
        survival=False,
        layer=1,
        rate=5,
        type='SRMamba',
        hidden_dim=256,
        local_layers=1,
        global_layers=1,
        pool_size=50,
        use_atp_pool=True,
        diffusion_steps=0,
        K_init=2.5,
        atp_dt=0.1,
        norm_type='mean',
        pool_mode='avg',
        tau_init=2.0,
        gamma_init=0.0,
        local_segment_mode='none',
        local_segment_size=50,
    )
    defaults.update(overrides)
    model = MambaMIL(**defaults)
    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# Test 1: Basic classification forward (avg pool, no segment)
# ============================================================
def test_basic_classification_forward():
    print("[Test 1] Basic classification forward (pool_mode=avg, local_segment_mode=none) ...")
    model = _make_model()
    x = torch.randn(1, 2500, 1024, device=DEVICE)
    out = model(x)

    assert len(out) == 5, f"Expected 5-tuple, got {len(out)}-tuple"
    logits, Y_prob, Y_hat, A_raw, results_dict = out

    assert logits.shape == (1, 2),  f"logits shape mismatch: {logits.shape}"
    assert Y_prob.shape == (1, 2),  f"Y_prob shape mismatch: {Y_prob.shape}"
    assert Y_hat.shape  == (1, 1),  f"Y_hat shape mismatch:  {Y_hat.shape}"
    assert results_dict is None,     f"results_dict should be None for classification"

    # Y_prob should sum to 1
    prob_sum = Y_prob.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones(1, device=DEVICE), atol=1e-5), \
        f"Y_prob does not sum to 1: {prob_sum}"

    # Y_hat should be valid class index
    assert (Y_hat >= 0).all() and (Y_hat < 2).all(), \
        f"Y_hat out of range: {Y_hat}"

    # No NaN
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    assert torch.isfinite(Y_prob).all(), "Y_prob contain NaN/Inf"

    print("  PASSED")


# ============================================================
# Test 2: Residual pool mode with backward
# ============================================================
def test_residual_pool_mode():
    print("[Test 2] Residual pool mode (pool_mode=residual, gamma_init=0.0) ...")
    model = _make_model(
        pool_mode='residual',
        gamma_init=0.0,
        tau_init=2.0,
    )
    x = torch.randn(1, 2500, 1024, device=DEVICE)
    out = model(x)

    logits = out[0]
    assert logits.shape == (1, 2), f"logits shape mismatch: {logits.shape}"
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf in residual mode"

    # Backward pass should succeed
    loss = logits.pow(2).mean()
    loss.backward()

    # Check gradients exist on ATPPool parameters
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

    print("  PASSED")


# ============================================================
# Test 3: Segment-wise Local Mamba (local_segment_mode=chunk)
# ============================================================
def test_segment_wise_chunk():
    print("[Test 3] Segment-wise Local Mamba (local_segment_mode=chunk) ...")
    model = _make_model(
        local_segment_mode='chunk',
        local_segment_size=50,
        pool_size=50,
    )
    x = torch.randn(1, 2500, 1024, device=DEVICE)
    out = model(x)

    logits, Y_prob, Y_hat, A_raw, _ = out
    assert logits.shape == (1, 2), f"logits shape mismatch: {logits.shape}"
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf in segment mode"

    print("  PASSED")


# ============================================================
# Test 4: No NaN in any output across modes
# ============================================================
def test_no_nan_across_modes():
    print("[Test 4] No NaN across pool modes ...")
    modes = [
        {'pool_mode': 'avg'},
        {'pool_mode': 'residual', 'gamma_init': 0.05, 'tau_init': 2.0},
    ]
    x = torch.randn(1, 2500, 1024, device=DEVICE)

    for cfg in modes:
        model = _make_model(**cfg)
        out = model(x)
        logits = out[0]
        assert torch.isfinite(logits).all(), \
            f"NaN/Inf detected with config {cfg}"
        print(f"  mode={cfg['pool_mode']}: OK")

    print("  PASSED")


# ============================================================
# Test 5: disable_atp_pool (Identity fallback)
# ============================================================
def test_disable_atp_pool():
    print("[Test 5] disable_atp_pool (use_atp_pool=False) ...")
    model = _make_model(use_atp_pool=False)
    x = torch.randn(1, 2500, 1024, device=DEVICE)
    out = model(x)
    logits = out[0]
    assert logits.shape == (1, 2), f"logits shape mismatch: {logits.shape}"
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf with atp_pool disabled"
    print("  PASSED")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MambaMIL Classification Forward Tests")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    test_basic_classification_forward()
    test_residual_pool_mode()
    test_segment_wise_chunk()
    test_no_nan_across_modes()
    test_disable_atp_pool()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
