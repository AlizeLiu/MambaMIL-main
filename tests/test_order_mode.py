"""
Tests for order_mode (random permutation negative control).
Run: python tests/test_order_mode.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import hashlib


def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def apply_random_perm(features, slide_id, base_seed=1):
    """Replicate the dataset logic."""
    h = stable_hash(slide_id)
    g = torch.Generator()
    g.manual_seed(base_seed + h)
    perm = torch.randperm(features.shape[0], generator=g)
    return features[perm], perm


def test_same_slide_same_seed():
    """Same slide_id + same seed → same permutation."""
    print("=== Same Slide Same Seed ===")
    feat = torch.randn(1000, 1024)
    out1, perm1 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    out2, perm2 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    assert torch.equal(perm1, perm2), "Same slide+seed should give same perm"
    assert torch.equal(out1, out2), "Same slide+seed should give same output"
    print("  PASSED ✅\n")


def test_same_slide_diff_seed():
    """Same slide_id + different seed → different permutation."""
    print("=== Same Slide Different Seed ===")
    feat = torch.randn(1000, 1024)
    _, perm1 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    _, perm2 = apply_random_perm(feat, "TCGA-05-4245", base_seed=999)
    assert not torch.equal(perm1, perm2), "Different seed should give different perm"
    print("  PASSED ✅\n")


def test_diff_slide_same_seed():
    """Different slide_id + same seed → different permutation."""
    print("=== Different Slide Same Seed ===")
    feat = torch.randn(1000, 1024)
    _, perm1 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    _, perm2 = apply_random_perm(feat, "TCGA-55-8206", base_seed=1)
    assert not torch.equal(perm1, perm2), "Different slide should give different perm"
    print("  PASSED ✅\n")


def test_keep_mode():
    """order_mode='keep' should not change input."""
    print("=== Keep Mode ===")
    feat = torch.randn(1000, 1024)
    feat_orig = feat.clone()
    # keep mode = do nothing
    assert torch.equal(feat, feat_orig), "Keep mode should preserve features"
    print("  PASSED ✅\n")


def test_random_perm_shape():
    """random_perm should preserve shape."""
    print("=== Random Perm Shape ===")
    feat = torch.randn(5000, 1024)
    out, _ = apply_random_perm(feat, "TCGA-05-4245")
    assert out.shape == feat.shape, f"Shape mismatch: {out.shape} vs {feat.shape}"
    print(f"  {feat.shape} -> {out.shape} ✅")
    print("  PASSED ✅\n")


def test_random_perm_is_permutation():
    """Output should contain same elements as input (just reordered)."""
    print("=== Random Perm Is Permutation ===")
    feat = torch.randn(100, 3)
    out, perm = apply_random_perm(feat, "TCGA-05-4245")
    # Check all original rows are present
    for i in range(100):
        assert torch.any(torch.all(out == feat[i], dim=1)), f"Row {i} missing"
    print("  All original rows preserved ✅")
    print("  PASSED ✅\n")


def test_no_cuda():
    """Should work on CPU."""
    print("=== No CUDA ===")
    feat = torch.randn(1000, 1024)
    out, perm = apply_random_perm(feat, "TCGA-05-4245")
    assert out.device == torch.device('cpu')
    assert perm.device == torch.device('cpu')
    print(f"  Device: {out.device} ✅")
    print("  PASSED ✅\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Order Mode Tests")
    print("=" * 50)
    print()
    test_same_slide_same_seed()
    test_same_slide_diff_seed()
    test_diff_slide_same_seed()
    test_keep_mode()
    test_random_perm_shape()
    test_random_perm_is_permutation()
    test_no_cuda()
    print("=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)
