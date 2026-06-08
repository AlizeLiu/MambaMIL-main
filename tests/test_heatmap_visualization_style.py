"""Tests for publication-quality heatmap visualization."""
import os
import sys
import numpy as np
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.heatmap_utils import (
    get_soft_pathology_colormap,
    get_colormap,
    normalize_attention,
    rasterize_attention,
    compute_tissue_outline,
    generate_topology_heatmap,
    generate_attention_panel_figure,
    compute_patch_attention,
)


def test_soft_pathology_colormap():
    """Test that soft_pathology returns LinearSegmentedColormap."""
    cmap = get_soft_pathology_colormap()
    from matplotlib.colors import LinearSegmentedColormap
    assert isinstance(cmap, LinearSegmentedColormap), f"Expected LinearSegmentedColormap, got {type(cmap)}"
    assert cmap.name == 'soft_pathology'
    print("[PASS] test_soft_pathology_colormap")


def test_get_colormap_styles():
    """Test various colormap styles."""
    cmap1 = get_colormap('soft_pathology')
    cmap2 = get_colormap('magma')
    cmap3 = get_colormap('viridis')
    cmap4 = get_colormap('jet_debug')
    
    assert cmap1 is not None
    assert cmap2 is not None
    assert cmap3 is not None
    assert cmap4 is not None
    print("[PASS] test_get_colormap_styles")


def test_no_hardcoded_jet():
    """Test that generate_topology_heatmap doesn't use hardcoded jet."""
    import inspect
    source = inspect.getsource(generate_topology_heatmap)
    
    # Should NOT have 'cmap = plt.cm.jet'
    assert 'cmap = plt.cm.jet' not in source, "generate_topology_heatmap still has hardcoded jet!"
    # Should use get_colormap
    assert 'get_colormap' in source, "generate_topology_heatmap should use get_colormap()"
    print("[PASS] test_no_hardcoded_jet")


def test_clip_percentile_effect():
    """Test that clip_percentile actually affects attention normalization."""
    np.random.seed(42)
    attn = np.random.exponential(0.1, 1000)
    attn[0] = 10.0  # outlier
    
    norm1 = normalize_attention(attn, low_clip=1, high_clip=99, gamma=1.0)
    norm2 = normalize_attention(attn, low_clip=1, high_clip=50, gamma=1.0)
    
    # Different clip percentiles should produce different results
    assert not np.allclose(norm1, norm2), "clip_percentile has no effect!"
    
    # Check no NaN
    assert not np.isnan(norm1).any(), "normalize_attention produced NaN!"
    assert not np.isnan(norm2).any(), "normalize_attention produced NaN!"
    print("[PASS] test_clip_percentile_effect")


def test_gamma_effect():
    """Test that gamma correction changes distribution."""
    np.random.seed(42)
    attn = np.random.uniform(0, 1, 1000)
    
    norm1 = normalize_attention(attn, gamma=1.0)
    norm2 = normalize_attention(attn, gamma=0.5)
    
    assert not np.allclose(norm1, norm2), "gamma has no effect!"
    print("[PASS] test_gamma_effect")


def test_rasterize_attention():
    """Test rasterize_attention output shape."""
    np.random.seed(42)
    coords = np.random.uniform(0, 10000, (500, 2))
    attn = np.random.uniform(0, 1, 500)
    
    canvas, extent = rasterize_attention(coords, attn, canvas_scale=32)
    
    assert canvas.ndim == 2, f"Canvas should be 2D, got {canvas.ndim}D"
    assert len(extent) == 4, f"Extent should have 4 values, got {len(extent)}"
    assert canvas.shape[0] > 0 and canvas.shape[1] > 0, "Canvas has zero dimension"
    print(f"[PASS] test_rasterize_attention: canvas shape={canvas.shape}")


def test_tissue_outline():
    """Test tissue outline computation."""
    np.random.seed(42)
    # Create a simple tissue-like distribution
    coords = np.column_stack([
        np.random.uniform(1000, 5000, 200),
        np.random.uniform(1000, 5000, 200),
    ])
    
    contours, extent, shape = compute_tissue_outline(coords, canvas_scale=32)
    
    assert isinstance(contours, list), "Contours should be a list"
    assert len(extent) == 4, "Extent should have 4 values"
    print(f"[PASS] test_tissue_outline: {len(contours)} contours found")


def test_panel_figure_saves_png():
    """Test that panel figure can save PNG."""
    np.random.seed(42)
    coords = np.column_stack([
        np.random.uniform(1000, 5000, 300),
        np.random.uniform(1000, 5000, 300),
    ])
    attn = np.random.uniform(0, 0.5, 300)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, 'test_panel.png')
        generate_attention_panel_figure(
            coords=coords,
            patch_attention=attn,
            slide_id='test_slide',
            output_path=out_path,
            pred=0, prob=[0.8, 0.2], label=1,
            dpi=72,  # low res for speed
        )
        assert os.path.exists(out_path), "PNG not saved!"
        assert os.path.getsize(out_path) > 1000, "PNG too small!"
    print("[PASS] test_panel_figure_saves_png")


def test_panel_figure_saves_pdf():
    """Test that panel figure can save PDF."""
    np.random.seed(42)
    coords = np.column_stack([
        np.random.uniform(1000, 5000, 200),
        np.random.uniform(1000, 5000, 200),
    ])
    attn = np.random.uniform(0, 0.5, 200)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, 'test_panel.png')
        generate_attention_panel_figure(
            coords=coords,
            patch_attention=attn,
            slide_id='test_slide',
            output_path=out_path,
            save_pdf=True,
            dpi=72,
        )
        pdf_path = out_path.replace('.png', '.pdf')
        assert os.path.exists(pdf_path), "PDF not saved!"
    print("[PASS] test_panel_figure_saves_pdf")


def test_attention_source_in_title():
    """Test that titles use 'Projected MIL Super-node Attention'."""
    import inspect
    source = inspect.getsource(generate_attention_panel_figure)
    
    assert 'Projected MIL Super-node Attention' in source, \
        "Panel figure should use 'Projected MIL Super-node Attention' as title"
    
    # Should NOT contain 'tumor probability'
    assert 'tumor probability' not in source.lower(), \
        "Panel figure should NOT reference 'tumor probability'"
    print("[PASS] test_attention_source_in_title")


def test_attention_mapping_in_csv():
    """Test that attention_mapping/projection_method is written to CSV."""
    # This tests the data flow in process_slide_heatmap
    import inspect
    source = inspect.getsource(sys.modules['utils.heatmap_utils'])
    
    assert 'attention_source' in source, "Should write 'attention_source' to CSV"
    assert 'projection_method' in source, "Should write 'projection_method' to CSV"
    print("[PASS] test_attention_mapping_in_csv")


def test_background_image_fallback():
    """Test that missing background image doesn't crash panel generation."""
    np.random.seed(42)
    coords = np.column_stack([
        np.random.uniform(1000, 5000, 200),
        np.random.uniform(1000, 5000, 200),
    ])
    attn = np.random.uniform(0, 0.5, 200)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, 'test_panel.png')
        # Pass non-existent background image path
        generate_attention_panel_figure(
            coords=coords,
            patch_attention=attn,
            slide_id='test_slide',
            output_path=out_path,
            background_image_path='/nonexistent/image.jpg',
            dpi=72,
        )
        assert os.path.exists(out_path), "Should fallback to tissue scatter!"
    print("[PASS] test_background_image_fallback")


def test_patch_attention_no_nan():
    """Test that compute_patch_attention returns no NaN."""
    np.random.seed(42)
    sn_attn = np.random.uniform(0, 1, 20)
    patch_attn = compute_patch_attention(sn_attn, n_patches=1000, pool_size=50)
    
    assert not np.isnan(patch_attn).any(), "patch_attention contains NaN!"
    assert len(patch_attn) == 1000, f"Expected 1000, got {len(patch_attn)}"
    print("[PASS] test_patch_attention_no_nan")


if __name__ == '__main__':
    test_soft_pathology_colormap()
    test_get_colormap_styles()
    test_no_hardcoded_jet()
    test_clip_percentile_effect()
    test_gamma_effect()
    test_rasterize_attention()
    test_tissue_outline()
    test_panel_figure_saves_png()
    test_panel_figure_saves_pdf()
    test_attention_source_in_title()
    test_attention_mapping_in_csv()
    test_background_image_fallback()
    test_patch_attention_no_nan()
    print(f"\n{'='*40}")
    print("  ALL 13 TESTS PASSED")
    print(f"{'='*40}")
