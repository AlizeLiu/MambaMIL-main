"""
Tests for heatmap_utils.py — topology-aware attention mapping.
Uses synthetic data, no real WSI dependencies.
"""
import os
import sys
import tempfile
import numpy as np
import torch
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.heatmap_utils import (
    load_coords_from_h5,
    load_hilbert_index,
    reorder_coords_by_hilbert,
    map_supernode_to_patches,
    compute_patch_attention,
    generate_topology_heatmap,
)


def _make_h5(path, coords):
    """Helper: write coords to an h5 file."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('coords', data=coords)


def _make_pt(path, idx):
    """Helper: write index tensor to a .pt file."""
    torch.save(torch.tensor(idx, dtype=torch.long), path)


# ------------------------------------------------------------------
# Test 1: Full no-sampling mapping
# ------------------------------------------------------------------
def test_full_no_sampling_mapping():
    """N=100, pool_size=10, M=10. Supernode i -> patches [10i, 10i+10).
    With assign mode (default): each patch gets supernode's attention value.
    """
    N, pool_size = 100, 10
    M = N // pool_size
    attn = np.arange(M, dtype=np.float32)  # [0, 1, 2, ..., 9]

    # assign mode (default)
    patch_attn = compute_patch_attention(attn, N, pool_size, mapping_mode='assign')
    assert patch_attn.shape == (N,), f"Expected shape ({N},), got {patch_attn.shape}"

    for m in range(M):
        start = m * pool_size
        end = start + pool_size
        expected_val = float(m)  # assign: same value
        assert np.allclose(patch_attn[start:end], expected_val), \
            f"Supernode {m}: expected {expected_val}, got {patch_attn[start:end]}"
    print("PASS: test_full_no_sampling_mapping (assign)")


def test_full_no_sampling_mapping_distribute():
    """Same setup but with distribute mode: patch_attn = supernode_attn / pool_size."""
    N, pool_size = 100, 10
    M = N // pool_size
    attn = np.arange(M, dtype=np.float32)

    patch_attn = compute_patch_attention(attn, N, pool_size, mapping_mode='distribute')
    for m in range(M):
        start = m * pool_size
        end = start + pool_size
        expected_val = float(m) / pool_size
        assert np.allclose(patch_attn[start:end], expected_val), \
            f"Supernode {m}: expected {expected_val}, got {patch_attn[start:end]}"
    print("PASS: test_full_no_sampling_mapping (distribute)")


# ------------------------------------------------------------------
# Test 2: Non-divisible N
# ------------------------------------------------------------------
def test_non_divisible_n():
    """N=105, pool_size=10 -> M=11 (last supernode has 5 patches)."""
    N, pool_size = 105, 10
    M = (N + pool_size - 1) // pool_size  # 11
    attn = np.arange(M, dtype=np.float32)

    patch_attn = compute_patch_attention(attn, N, pool_size)
    assert patch_attn.shape == (N,)

    # Last supernode (idx=10) -> patches 100-104, value=10 (assign mode)
    assert np.allclose(patch_attn[100:105], 10.0), \
        f"Last supernode patches wrong: {patch_attn[100:105]}"
    print("PASS: test_non_divisible_n")


# ------------------------------------------------------------------
# Test 3: Sampled indices mapping
# ------------------------------------------------------------------
def test_sampled_indices_mapping():
    """coords N=1000, sampled_indices=[0..99], pool_size=10."""
    N_total = 1000
    sampled = np.arange(100)
    pool_size = 10
    M = len(sampled) // pool_size
    attn = np.arange(M, dtype=np.float32) + 1.0

    # compute_patch_attention works on the sampled sequence length
    patch_attn = compute_patch_attention(attn, len(sampled), pool_size)
    assert patch_attn.shape == (100,)
    # First supernode -> patches 0-9, value=1.0 (assign mode)
    assert np.allclose(patch_attn[:10], 1.0)
    print("PASS: test_sampled_indices_mapping")


# ------------------------------------------------------------------
# Test 4: Hilbert index coords reorder
# ------------------------------------------------------------------
def test_hilbert_index_reorder():
    """coords raw = [[0,0],[1,1],[2,2],[3,3]], idx = [3,2,1,0] (reversed)."""
    coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
    idx = np.array([3, 2, 1, 0])

    coords_h = reorder_coords_by_hilbert(coords, idx)
    expected = np.array([[3, 3], [2, 2], [1, 1], [0, 0]], dtype=np.float32)
    assert np.allclose(coords_h, expected), f"Got {coords_h}"
    print("PASS: test_hilbert_index_reorder")


# ------------------------------------------------------------------
# Test 5: Scatter heatmap output
# ------------------------------------------------------------------
def test_scatter_heatmap_output():
    """Generate a scatter heatmap PNG and verify it exists and is non-empty."""
    coords = np.random.rand(50, 2) * 1000
    patch_attn = np.random.rand(50).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_heatmap.png")
        generate_topology_heatmap(
            coords=coords,
            patch_attention=patch_attn,
            slide_id="TEST_SLIDE",
            save_path=save_path,
        )
        assert os.path.exists(save_path), f"PNG not created: {save_path}"
        size = os.path.getsize(save_path)
        assert size > 1000, f"PNG too small ({size} bytes), likely empty"
    print("PASS: test_scatter_heatmap_output")


# ------------------------------------------------------------------
# Test 6: load_coords_from_h5
# ------------------------------------------------------------------
def test_load_coords_from_h5():
    """Test h5 coords loading."""
    coords = np.random.randint(0, 1000, size=(50, 2)).astype(np.int32)
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test.h5")
        _make_h5(h5_path, coords)
        loaded = load_coords_from_h5(h5_path)
        assert loaded.shape == (50, 2)
        assert np.array_equal(loaded, coords)
    print("PASS: test_load_coords_from_h5")


# ------------------------------------------------------------------
# Test 7: load_hilbert_index
# ------------------------------------------------------------------
def test_load_hilbert_index():
    """Test hilbert index loading from .pt file."""
    idx = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = os.path.join(tmpdir, "test_hilbert.pt")
        _make_pt(pt_path, idx)
        loaded = load_hilbert_index(pt_path)
        assert np.array_equal(loaded, idx)
    print("PASS: test_load_hilbert_index")


# ------------------------------------------------------------------
# Test 8: map_supernode_to_patches
# ------------------------------------------------------------------
def test_map_supernode_to_patches():
    """Test supernode -> patch range mapping."""
    # N=100, pool_size=10
    patches = map_supernode_to_patches(100, 10, 0)
    assert patches == list(range(0, 10)), f"Got {patches}"

    patches = map_supernode_to_patches(100, 10, 9)
    assert patches == list(range(90, 100)), f"Got {patches}"

    # Non-divisible: N=105, pool_size=10, last supernode
    patches = map_supernode_to_patches(105, 10, 10)
    assert patches == list(range(100, 105)), f"Got {patches}"
    print("PASS: test_map_supernode_to_patches")


# ------------------------------------------------------------------
# Test 9: Error on missing coords
# ------------------------------------------------------------------
def test_error_on_missing_coords():
    """load_coords_from_h5 should raise FileNotFoundError for missing file."""
    try:
        load_coords_from_h5("/nonexistent/path.h5")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    print("PASS: test_error_on_missing_coords")


# ------------------------------------------------------------------
# Test 10: Error on missing coords key
# ------------------------------------------------------------------
def test_error_on_missing_coords_key():
    """load_coords_from_h5 should raise KeyError if 'coords' not in h5."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "bad.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('features', data=np.zeros((10, 1024)))
        try:
            load_coords_from_h5(h5_path)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass
    print("PASS: test_error_on_missing_coords_key")


# ------------------------------------------------------------------
if __name__ == '__main__':
    test_full_no_sampling_mapping()
    test_full_no_sampling_mapping_distribute()
    test_non_divisible_n()
    test_sampled_indices_mapping()
    test_hilbert_index_reorder()
    test_scatter_heatmap_output()
    test_load_coords_from_h5()
    test_load_hilbert_index()
    test_map_supernode_to_patches()
    test_error_on_missing_coords()
    test_error_on_missing_coords_key()
    print("\n=== ALL HEATMAP TESTS PASSED ===")
