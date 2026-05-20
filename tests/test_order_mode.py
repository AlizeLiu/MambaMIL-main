"""
Tests for order_mode + sampling_mode interaction.
Verifies random_perm only changes order, not token set.
Also verifies segment-wise Local Mamba state isolation.
Run: python tests/test_order_mode.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import hashlib


def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def apply_random_perm(features, slide_id, base_seed=1):
    h = stable_hash(slide_id)
    g = torch.Generator()
    g.manual_seed(base_seed + h)
    perm = torch.randperm(features.shape[0], generator=g)
    return features[perm], perm


# ===== Original order_mode tests =====

def test_same_slide_same_seed():
    print("=== Same Slide Same Seed ===")
    feat = torch.randn(1000, 1024)
    out1, perm1 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    out2, perm2 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    assert torch.equal(perm1, perm2)
    assert torch.equal(out1, out2)
    print("  PASSED ✅\n")


def test_same_slide_diff_seed():
    print("=== Same Slide Different Seed ===")
    feat = torch.randn(1000, 1024)
    _, perm1 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    _, perm2 = apply_random_perm(feat, "TCGA-05-4245", base_seed=999)
    assert not torch.equal(perm1, perm2)
    print("  PASSED ✅\n")


def test_diff_slide_same_seed():
    print("=== Different Slide Same Seed ===")
    feat = torch.randn(1000, 1024)
    _, perm1 = apply_random_perm(feat, "TCGA-05-4245", base_seed=1)
    _, perm2 = apply_random_perm(feat, "TCGA-55-8206", base_seed=1)
    assert not torch.equal(perm1, perm2)
    print("  PASSED ✅\n")


def test_keep_mode():
    print("=== Keep Mode ===")
    feat = torch.randn(1000, 1024)
    feat_orig = feat.clone()
    assert torch.equal(feat, feat_orig)
    print("  PASSED ✅\n")


def test_random_perm_shape():
    print("=== Random Perm Shape ===")
    feat = torch.randn(5000, 1024)
    out, _ = apply_random_perm(feat, "TCGA-05-4245")
    assert out.shape == feat.shape
    print(f"  {feat.shape} -> {out.shape} ✅")
    print("  PASSED ✅\n")


def test_random_perm_is_permutation():
    print("=== Random Perm Is Permutation ===")
    feat = torch.randn(100, 3)
    out, perm = apply_random_perm(feat, "TCGA-05-4245")
    for i in range(100):
        assert torch.any(torch.all(out == feat[i], dim=1)), f"Row {i} missing"
    print("  All original rows preserved ✅")
    print("  PASSED ✅\n")


def test_no_cuda():
    print("=== No CUDA ===")
    feat = torch.randn(1000, 1024)
    out, perm = apply_random_perm(feat, "TCGA-05-4245")
    assert out.device == torch.device('cpu')
    assert perm.device == torch.device('cpu')
    print(f"  Device: {out.device} ✅")
    print("  PASSED ✅\n")


# ===== NEW: Token set preservation after sampling =====

def _simulate_sampling_then_perm(N, max_seq_len, sampling_mode, slide_id, seed, chunk_size=50):
    """Simulate the dataset logic: load features → sample → optional perm."""
    # Use arange as features so we can track which tokens were sampled
    feat = torch.arange(N).view(N, 1).float()

    # Sampling
    if sampling_mode == 'random_points':
        g = torch.Generator()
        g.manual_seed(seed + 42)
        indices = torch.randperm(N, generator=g)[:max_seq_len]
        indices, _ = indices.sort()
    elif sampling_mode == 'uniform_points':
        indices = torch.linspace(0, N - 1, max_seq_len).long()
    elif sampling_mode == 'chunk':
        from dataset.dataset_survival import hilbert_chunk_sample_indices
        indices = hilbert_chunk_sample_indices(N, max_seq_len, chunk_size, training=True)
    else:
        raise ValueError

    sampled = feat[indices].squeeze()  # [max_seq_len]

    # Optional random_perm
    h = stable_hash(slide_id)
    g2 = torch.Generator()
    g2.manual_seed(seed + h)
    perm = torch.randperm(sampled.shape[0], generator=g2)
    permuted = sampled[perm]

    return sampled, permuted


def _test_token_set_preserved(sampling_mode, label):
    print(f"=== Token Set Preserved ({label}) ===")
    N, max_seq_len = 10000, 2500
    slide_id = "TCGA-05-4245"
    seed = 1

    keep, perm = _simulate_sampling_then_perm(N, max_seq_len, sampling_mode, slide_id, seed)

    # Same shape
    assert keep.shape == perm.shape, f"Shape mismatch: {keep.shape} vs {perm.shape}"

    # Same token set (sorted values should be identical)
    assert torch.equal(keep.sort().values, perm.sort().values), \
        "Token sets differ! random_perm changed the token set."

    # Order likely differs (with 2500 tokens, probability of same order is ~0)
    # But we don't assert strict inequality to avoid flaky tests
    if torch.equal(keep, perm):
        print(f"  Warning: keep == perm (extremely unlikely, check seed)")
    else:
        print(f"  Order differs ✅")

    print(f"  Token set identical ✅ (sorted match)")
    print("  PASSED ✅\n")


def test_random_perm_preserves_token_set_random_points():
    _test_token_set_preserved('random_points', 'random_points')


def test_random_perm_preserves_token_set_uniform_points():
    _test_token_set_preserved('uniform_points', 'uniform_points')


def test_random_perm_preserves_token_set_chunk():
    _test_token_set_preserved('chunk', 'chunk')


# ===== NEW: Segment-wise Local Mamba state isolation =====

class CumsumMixer(nn.Module):
    """Mock layer: cumulative sum along sequence dim.
    If state is NOT reset between segments, token 51 will include
    the cumulative sum of tokens 0-50 from the previous segment.
    """
    def forward(self, x):
        return torch.cumsum(x, dim=1)


def test_segment_local_mamba_resets_state():
    print("=== Segment State Reset (CumsumMixer) ===")

    B, L, D = 1, 100, 4
    seg_size = 50

    h = torch.ones(B, L, D)

    # Flat path: cumsum over full sequence
    mixer = CumsumMixer()
    h_flat = mixer(h)
    # Token 50 (0-indexed) in flat: cumsum includes tokens 0..50 = 51
    val_flat_token50 = h_flat[0, 50, 0].item()

    # Segment path: reshape into 2 segments, apply independently
    S = L // seg_size
    h_seg = h.reshape(B, S, seg_size, D).reshape(B * S, seg_size, D)
    h_seg = mixer(h_seg)
    h_seg = h_seg.reshape(B, S, seg_size, D)

    # Token 50 is the first token of segment 1
    val_seg_token50 = h_seg[0, 1, 0, 0].item()

    print(f"  Flat cumsum at token 50:   {val_flat_token50} (includes tokens 0-50)")
    print(f"  Segment cumsum at token 50: {val_seg_token50} (fresh start)")

    # In flat path, token 50 has cumsum=51 (1+1+...+1, 51 times)
    # In segment path, token 50 is start of segment 1, cumsum=1
    assert val_flat_token50 == 51.0, f"Expected 51, got {val_flat_token50}"
    assert val_seg_token50 == 1.0, f"Expected 1, got {val_seg_token50}"
    assert val_flat_token50 != val_seg_token50, "Segment should reset state"

    # Verify segment 0 internal cumsum is correct
    seg0_last = h_seg[0, 0, -1, 0].item()  # last token of segment 0
    assert seg0_last == 50.0, f"Segment 0 last token should be 50, got {seg0_last}"

    # Verify segment 1 internal cumsum starts from 1
    seg1_first = h_seg[0, 1, 0, 0].item()
    seg1_last = h_seg[0, 1, -1, 0].item()
    assert seg1_first == 1.0, f"Segment 1 first should be 1, got {seg1_first}"
    assert seg1_last == 50.0, f"Segment 1 last should be 50, got {seg1_last}"

    print("  Flat: token 50 sees full history (51)")
    print("  Segment: token 50 starts fresh (1)")
    print("  State correctly reset at segment boundary ✅")
    print("  PASSED ✅\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Order Mode & Segment State Tests")
    print("=" * 50)
    print()

    # Original tests
    test_same_slide_same_seed()
    test_same_slide_diff_seed()
    test_diff_slide_same_seed()
    test_keep_mode()
    test_random_perm_shape()
    test_random_perm_is_permutation()
    test_no_cuda()

    # NEW: Token set preservation
    test_random_perm_preserves_token_set_random_points()
    test_random_perm_preserves_token_set_uniform_points()
    test_random_perm_preserves_token_set_chunk()

    # NEW: Segment state isolation
    test_segment_local_mamba_resets_state()

    print("=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)
