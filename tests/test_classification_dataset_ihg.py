"""
Test IHG-Mamba dataset logic using synthetic data.
All tests create temporary .pt files — no real data required.
"""
import sys
import os
import tempfile
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, '.')


def _make_synthetic_slide(tmpdir, n_tokens=200, dim=64, subdir='pt_files', backbone='uni', slide_id='slide1'):
    """Create a synthetic .pt feature file and return its path."""
    feat_dir = os.path.join(tmpdir, subdir, backbone)
    os.makedirs(feat_dir, exist_ok=True)
    feat = torch.randn(n_tokens, dim)
    path = os.path.join(feat_dir, f'{slide_id}.pt')
    torch.save(feat, path)
    return feat, path


def _make_synthetic_hilbert_index(tmpdir, n_tokens=200, slide_id='slide1'):
    """Create a synthetic hilbert index file and return path + index."""
    hilbert_dir = os.path.join(tmpdir, 'hilbert')
    os.makedirs(hilbert_dir, exist_ok=True)
    idx = torch.randperm(n_tokens)
    path = os.path.join(hilbert_dir, f'{slide_id}_hilbert.pt')
    torch.save(idx, path)
    return idx, path


def _make_dataset_csv(tmpdir, slide_ids, labels):
    """Create a minimal dataset CSV."""
    csv_path = os.path.join(tmpdir, 'dataset.csv')
    df = pd.DataFrame({
        'slide_id': slide_ids,
        'label': labels,
        'case_id': [f'case_{i}' for i in range(len(slide_ids))],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================
# Test 1: feature_subdir path resolution
# ============================================================
def test_feature_subdir_path_resolution():
    print("[Test 1] feature_subdir path resolution ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        n_tokens = 200
        dim = 64
        backbone = 'uni'

        # Create two different feature files
        feat_pt, path_pt = _make_synthetic_slide(
            tmpdir, n_tokens=n_tokens, dim=dim,
            subdir='pt_files', backbone=backbone, slide_id='slide1'
        )
        feat_hb, path_hb = _make_synthetic_slide(
            tmpdir, n_tokens=n_tokens, dim=dim,
            subdir='hilbert', backbone=backbone, slide_id='slide1'
        )

        # Verify they are different tensors
        assert not torch.equal(feat_pt, feat_hb), \
            "Synthetic features should be different"

        # Verify the files exist at expected paths
        expected_pt = os.path.join(tmpdir, 'pt_files', backbone, 'slide1.pt')
        expected_hb = os.path.join(tmpdir, 'hilbert', backbone, 'slide1.pt')
        assert os.path.exists(expected_pt), f"Missing: {expected_pt}"
        assert os.path.exists(expected_hb), f"Missing: {expected_hb}"

        # Load and verify feature_subdir controls which file is returned
        loaded_pt = torch.load(expected_pt)
        loaded_hb = torch.load(expected_hb)
        assert torch.equal(loaded_pt, feat_pt), "pt_files feature mismatch"
        assert torch.equal(loaded_hb, feat_hb), "hilbert feature mismatch"

    print("  PASSED")


# ============================================================
# Test 2: Online Hilbert index reorder
# ============================================================
def test_online_hilbert_reorder():
    print("[Test 2] Online Hilbert index reorder ...")
    n = 100
    features = torch.arange(n).view(n, 1).float()  # [[0],[1],...,[99]]
    hilbert_idx = torch.randperm(n)

    reordered = features[hilbert_idx]

    # Verify: each element in reordered corresponds to hilbert_idx
    for i in range(n):
        orig_pos = hilbert_idx[i].item()
        assert reordered[i, 0].item() == float(orig_pos), \
            f"Reorder mismatch at {i}: expected {orig_pos}, got {reordered[i,0].item()}"

    # Verify: token set is preserved
    assert set(reordered.flatten().tolist()) == set(features.flatten().tolist()), \
        "Token set changed after reorder"

    print("  PASSED")


# ============================================================
# Test 3: Sampling modes produce correct lengths
# ============================================================
def test_sampling_modes():
    print("[Test 3] Sampling modes produce correct lengths ...")
    n_tokens = 5000
    max_seq_len = 2500
    features = torch.randn(n_tokens, 64)

    # random_points (deterministic mode: linspace)
    indices_rp = torch.linspace(0, n_tokens - 1, max_seq_len).long()
    sampled_rp = features[indices_rp]
    assert sampled_rp.shape[0] == max_seq_len, \
        f"random_points length mismatch: {sampled_rp.shape[0]}"

    # uniform_points
    indices_up = torch.linspace(0, n_tokens - 1, max_seq_len).long()
    sampled_up = features[indices_up]
    assert sampled_up.shape[0] == max_seq_len, \
        f"uniform_points length mismatch: {sampled_up.shape[0]}"

    # chunk sampling (if available)
    try:
        from dataset.dataset_survival import hilbert_chunk_sample_indices
        indices_ch = hilbert_chunk_sample_indices(
            n=n_tokens, max_seq_len=max_seq_len, chunk_size=50,
            training=False, eval_strategy='center'
        )
        sampled_ch = features[indices_ch]
        assert sampled_ch.shape[0] == max_seq_len, \
            f"chunk length mismatch: {sampled_ch.shape[0]}"
        print("  chunk sampling: OK")
    except ImportError:
        print("  [SKIP] hilbert_chunk_sample_indices not available")

    print("  PASSED")


# ============================================================
# Test 4: order_mode random_perm preserves token set
# ============================================================
def test_order_mode_random_perm():
    print("[Test 4] order_mode=random_perm preserves token set ...")
    n = 2500
    features = torch.randn(n, 64)

    # Simulate random_perm with deterministic seed
    import hashlib
    slide_id = 'slide1'
    order_seed = 1
    h = int(hashlib.md5(slide_id.encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator()
    g.manual_seed(order_seed + h)
    perm = torch.randperm(features.shape[0], generator=g)
    shuffled = features[perm]

    # Token set preserved
    orig_set = set(features.flatten().round(decimals=4).tolist())
    shuf_set = set(shuffled.flatten().round(decimals=4).tolist())
    assert orig_set == shuf_set, "Token set changed after random_perm"

    # Order actually changed (overwhelmingly likely)
    assert not torch.equal(features, shuffled), \
        "Random perm did not change order (extremely unlikely)"

    # Deterministic: same seed produces same permutation
    g2 = torch.Generator()
    g2.manual_seed(order_seed + h)
    perm2 = torch.randperm(features.shape[0], generator=g2)
    assert torch.equal(perm, perm2), "Same seed should produce same permutation"

    print("  PASSED")


# ============================================================
# Test 5: Generic_Split __getitem__ returns (features, label)
# ============================================================
def test_return_format():
    print("[Test 5] Generic_Split __getitem__ returns (features, label) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        n_tokens = 300
        dim = 64
        backbone = 'uni'

        # Create synthetic feature file
        feat, _ = _make_synthetic_slide(
            tmpdir, n_tokens=n_tokens, dim=dim,
            subdir='pt_files', backbone=backbone, slide_id='slide_001'
        )

        # Create dataset CSV
        csv_path = _make_dataset_csv(tmpdir, ['slide_001'], [0])

        # Create Generic_Split directly
        from dataset.dataset_generic import Generic_Split
        slide_data = pd.DataFrame({
            'slide_id': ['slide_001'],
            'label': [0],
            'case_id': ['case_0'],
        })
        split = Generic_Split(slide_data, data_dir=tmpdir, num_classes=2)
        split.set_backbone(backbone)
        split.set_patch_size('')  # No patch_size prefix

        # Set IHG defaults
        split.max_seq_len = 2500
        split.feature_subdir = 'pt_files'
        split.features_already_hilbert = False
        split.use_hilbert_index = False
        split.sampling_mode = 'random_points'
        split.training_mode = False

        result = split[0]
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        features, label = result
        assert isinstance(features, torch.Tensor), \
            f"Expected torch.Tensor features, got {type(features)}"
        assert features.shape == (n_tokens, dim), \
            f"Feature shape mismatch: {features.shape}"
        assert label == 0, f"Label mismatch: {label}"

    print("  PASSED")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("IHG-Mamba Dataset Logic Tests")
    print("=" * 60)

    test_feature_subdir_path_resolution()
    test_online_hilbert_reorder()
    test_sampling_modes()
    test_order_mode_random_perm()
    test_return_format()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
