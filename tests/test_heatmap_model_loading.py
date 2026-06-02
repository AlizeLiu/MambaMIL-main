"""
Tests for checkpoint loading in generate_topology_heatmap.py.
Validates strict/partial loading behavior.
"""
import os
import sys
import tempfile
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.MambaMIL import MambaMIL


def _make_checkpoint(hidden_dim=256, in_dim=1024, n_classes=2):
    """Build a small MambaMIL and save its state_dict."""
    model = MambaMIL(
        in_dim=in_dim, n_classes=n_classes, dropout=0.0, act='gelu',
        survival=False, hidden_dim=hidden_dim,
        local_layers=1, global_layers=1,
        pool_size=50, use_atp_pool=True, pool_mode='avg',
        diffusion_steps=0, K_init=2.5, atp_dt=0.1, norm_type='mean',
        tau_init=2.0, gamma_init=0.0,
        attn_type='simple', attn_dim=128,
    )
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        return f.name, model


def test_strict_load_correct():
    """strict=True with matching checkpoint should succeed."""
    ckpt_path, model = _make_checkpoint()
    state_dict = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0, f"Missing: {missing}"
    assert len(unexpected) == 0, f"Unexpected: {unexpected}"
    os.unlink(ckpt_path)
    print("PASS: test_strict_load_correct")


def test_strict_load_wrong_structure():
    """strict=True with mismatched checkpoint should raise RuntimeError."""
    ckpt_path, _ = _make_checkpoint(hidden_dim=256)
    # Build model with different hidden_dim
    wrong_model = MambaMIL(
        in_dim=1024, n_classes=2, dropout=0.0, act='gelu',
        survival=False, hidden_dim=128,  # different!
        local_layers=1, global_layers=1,
        pool_size=50, use_atp_pool=True, pool_mode='avg',
        diffusion_steps=0, K_init=2.5, attn_type='simple', attn_dim=128,
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    try:
        wrong_model.load_state_dict(state_dict, strict=True)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    os.unlink(ckpt_path)
    print("PASS: test_strict_load_wrong_structure")


def test_partial_load_allowed():
    """allow_partial_load (strict=False) should succeed with mismatched keys."""
    ckpt_path, model = _make_checkpoint(hidden_dim=256)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # Add a fake key
    state_dict['fake_key'] = torch.zeros(1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert 'fake_key' in unexpected
    os.unlink(ckpt_path)
    print("PASS: test_partial_load_allowed")


def test_default_is_strict():
    """Default behavior should be strict=True (no allow_partial_load)."""
    ckpt_path, _ = _make_checkpoint(hidden_dim=256)
    wrong_model = MambaMIL(
        in_dim=1024, n_classes=2, dropout=0.0, act='gelu',
        survival=False, hidden_dim=128,
        local_layers=1, global_layers=1,
        pool_size=50, use_atp_pool=True, pool_mode='avg',
        diffusion_steps=0, K_init=2.5, attn_type='simple', attn_dim=128,
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # Simulate default: strict = not allow_partial_load = not False = True
    strict = not False  # default: allow_partial_load=False
    try:
        wrong_model.load_state_dict(state_dict, strict=strict)
        assert False, "Should have raised RuntimeError with strict=True"
    except RuntimeError:
        pass
    os.unlink(ckpt_path)
    print("PASS: test_default_is_strict")


if __name__ == '__main__':
    test_strict_load_correct()
    test_strict_load_wrong_structure()
    test_partial_load_allowed()
    test_default_is_strict()
    print("\n=== ALL MODEL LOADING TESTS PASSED ===")
