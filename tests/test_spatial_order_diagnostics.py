"""
Tests for tools/analyze_spatial_order.py — spatial continuity diagnostics.
Uses synthetic grid coords.
"""
import os
import sys
import tempfile
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.analyze_spatial_order import (
    apply_order,
    apply_sampling,
    compute_jump_distances,
    compute_tear_rate,
    compute_pool_window_diameters,
    compute_coverage_ratio,
    analyze_slide,
)


def _make_grid(n=32):
    """Create a regular grid of n*n points."""
    x = np.arange(n, dtype=np.float64)
    y = np.arange(n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def _make_hilbert_idx(n):
    """Mock Hilbert index: reversed order."""
    return np.arange(n)[::-1]


# ------------------------------------------------------------------
# Test 1: Raster jump < random_perm jump
# ------------------------------------------------------------------
def test_raster_less_than_random():
    """Raster-ordered grid should have smaller jumps than random permutation."""
    coords = _make_grid(20)  # 400 points
    raster = apply_order(coords, 'raw')
    random = apply_order(coords, 'random_perm', seed=42)

    raster_jumps = compute_jump_distances(raster)
    random_jumps = compute_jump_distances(random)

    assert np.mean(raster_jumps) < np.mean(random_jumps), \
        f"Raster mean={np.mean(raster_jumps):.2f} should be < random mean={np.mean(random_jumps):.2f}"
    print(f"PASS: raster ({np.mean(raster_jumps):.2f}) < random ({np.mean(random_jumps):.2f})")


# ------------------------------------------------------------------
# Test 2: Chunk sampling reduces internal jump distance
# ------------------------------------------------------------------
def test_chunk_internal_jumps_small():
    """Chunk-sampled coords should have small internal jumps within chunks."""
    coords = _make_grid(20)
    hilbert_idx = np.arange(len(coords))  # identity for simplicity
    coords_h = apply_order(coords, 'hilbert', hilbert_idx)

    _, idx = apply_sampling(coords_h, 'chunk', max_seq_len=100, chunk_size=10)
    sampled = coords_h[idx]

    # Compute jumps
    jumps = compute_jump_distances(sampled)
    # Most jumps should be small (within-chunk)
    median_jump = np.median(jumps)
    assert median_jump < 5.0, f"Median jump too large: {median_jump:.2f}"
    print(f"PASS: chunk median jump = {median_jump:.2f}")


# ------------------------------------------------------------------
# Test 3: Coverage ratio
# ------------------------------------------------------------------
def test_coverage_ratio():
    """Sampled subset should have coverage < 1.0."""
    coords = _make_grid(32)
    sampled = coords[:100]  # 100 out of 1024

    cov = compute_coverage_ratio(coords, sampled, grid_size=16)
    assert 0 < cov < 1.0, f"Coverage should be between 0 and 1, got {cov}"
    print(f"PASS: coverage_ratio = {cov:.4f}")


# ------------------------------------------------------------------
# Test 4: Pool window diameter
# ------------------------------------------------------------------
def test_pool_window_diameter():
    """Pool windows on a grid should have predictable diameters."""
    coords = _make_grid(10)  # 100 points, 10x10 grid
    diams = compute_pool_window_diameters(coords, pool_size=10)
    assert len(diams) == 10
    # Each window has 10 consecutive points on a 10x10 grid
    # Diameters should be non-negative
    assert all(d >= 0 for d in diams)
    print(f"PASS: pool_window_diameter, mean={np.mean(diams):.2f}")


# ------------------------------------------------------------------
# Test 5: analyze_slide returns correct fields
# ------------------------------------------------------------------
def test_analyze_slide_fields():
    """analyze_slide should return all expected diagnostic fields."""
    coords = _make_grid(20)
    result, _, _, _ = analyze_slide(
        'test_slide', coords, order_mode='raw', sampling_mode='none'
    )
    required = [
        'slide_id', 'order_mode', 'sampling_mode', 'n_tokens_full', 'n_tokens_sampled',
        'mean_jump', 'median_jump', 'p90_jump', 'p99_jump', 'max_jump',
        'tear_rate', 'mean_window_diameter', 'coverage_ratio',
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"
    assert result['slide_id'] == 'test_slide'
    assert result['n_tokens_full'] == 400
    assert result['n_tokens_sampled'] == 400
    print("PASS: analyze_slide_fields")


# ------------------------------------------------------------------
# Test 6: No NaN in output
# ------------------------------------------------------------------
def test_no_nan():
    """No NaN values in diagnostics output."""
    coords = _make_grid(20)
    result, _, _, _ = analyze_slide(
        'test', coords, order_mode='hilbert',
        hilbert_idx=_make_hilbert_idx(400),
        sampling_mode='random_points', max_seq_len=100
    )
    for k, v in result.items():
        if isinstance(v, float):
            assert not np.isnan(v), f"NaN in {k}"
    print("PASS: no_nan")


# ------------------------------------------------------------------
# Test 7: Output CSV
# ------------------------------------------------------------------
def test_output_csv():
    """analyze_slide + save should produce valid CSV."""
    import pandas as pd
    coords = _make_grid(10)
    result, _, _, _ = analyze_slide('test', coords, order_mode='raw')

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'test.csv')
        pd.DataFrame([result]).to_csv(csv_path, index=False)
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert 'slide_id' in df.columns
        assert 'mean_jump' in df.columns
    print("PASS: output_csv")


# ------------------------------------------------------------------
if __name__ == '__main__':
    test_raster_less_than_random()
    test_chunk_internal_jumps_small()
    test_coverage_ratio()
    test_pool_window_diameter()
    test_analyze_slide_fields()
    test_no_nan()
    test_output_csv()
    print("\n=== ALL SPATIAL ORDER DIAGNOSTICS TESTS PASSED ===")
