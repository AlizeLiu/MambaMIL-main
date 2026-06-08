"""
Heatmap utilities for topology-aware attention visualization.
Maps super-node attention weights to patch-level coordinates using Hilbert curve ordering.

Publication-quality visualization:
- Panel figure (tissue map / outline / attention heatmap)
- Raster canvas rendering
- Soft pathology colormap
- Gamma-corrected attention normalization
- Alignment verification
- Tissue/background quality control
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


# ============================================================
# Colormaps
# ============================================================

def get_soft_pathology_colormap():
    """Soft pathology colormap: near-white -> pale yellow -> muted orange -> deep red.
    Designed for publication heatmaps with white background."""
    colors = [
        "#f7f7f7",  # very low, near white
        "#fff7bc",  # pale yellow
        "#fee391",  # light yellow-orange
        "#fdae61",  # muted orange
        "#e6550d",  # muted red-orange
        "#a63603",  # deep muted red
    ]
    rgb_colors = []
    for hex_color in colors:
        h = hex_color.lstrip('#')
        rgb_colors.append(tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)))
    return LinearSegmentedColormap.from_list('soft_pathology', rgb_colors, N=256)


def get_colormap(style='soft_pathology'):
    """Get colormap by style name.
    
    Args:
        style: 'soft_pathology', 'magma', 'YlOrRd', 'viridis', 'jet_debug'
    
    Returns:
        matplotlib colormap
    """
    if style == 'soft_pathology':
        return get_soft_pathology_colormap()
    elif style == 'magma':
        return plt.cm.magma
    elif style in ('YlOrRd', 'yl_or_rd'):
        return plt.cm.YlOrRd
    elif style == 'viridis':
        return plt.cm.viridis
    elif style == 'jet_debug':
        return plt.cm.jet
    elif style == 'jet':
        warnings.warn("[WARNING] jet is not recommended for publication heatmaps. Use 'soft_pathology' instead.")
        return plt.cm.jet
    else:
        try:
            return plt.get_cmap(style)
        except ValueError:
            warnings.warn(f"[WARNING] Unknown cmap '{style}', falling back to soft_pathology")
            return get_soft_pathology_colormap()


# ============================================================
# Attention normalization (with clipping + gamma)
# ============================================================

def normalize_attention(patch_attention, low_clip=1, high_clip=99, gamma=0.7):
    """Normalize attention with percentile clipping and gamma correction.
    
    Args:
        patch_attention: np.ndarray [N], raw attention values
        low_clip: lower percentile for clipping (0-100)
        high_clip: upper percentile for clipping (0-100)
        gamma: gamma correction exponent (0-1, lower = more contrast)
    
    Returns:
        attn_norm: np.ndarray [N], normalized to [0, 1], gamma-corrected
    """
    vmin = np.percentile(patch_attention, low_clip)
    vmax = np.percentile(patch_attention, high_clip)
    
    attn_clip = np.clip(patch_attention, vmin, vmax)
    
    if vmax > vmin:
        attn_norm = (attn_clip - vmin) / (vmax - vmin + 1e-8)
    else:
        attn_norm = np.zeros_like(patch_attention)
    
    # Gamma correction for better contrast
    attn_norm = np.power(attn_norm, gamma)
    
    # Ensure no NaN
    attn_norm = np.nan_to_num(attn_norm, nan=0.0, posinf=1.0, neginf=0.0)
    
    return attn_norm


# ============================================================
# Raster canvas rendering
# ============================================================

def rasterize_attention(coords, attention, patch_size=512, canvas_scale=32, agg='max'):
    """Rasterize coords and attention into a 2D canvas for heatmap display.
    
    Args:
        coords: np.ndarray [N, 2], patch coordinates
        attention: np.ndarray [N], attention values (normalized 0-1)
        patch_size: original patch size (for cell size calculation)
        canvas_scale: downscale factor (e.g. 32 means each cell = 32x32 pixels)
        agg: aggregation mode 'max' or 'mean'
    
    Returns:
        canvas: np.ndarray [H, W], attention heatmap (NaN for empty cells)
        extent: [x_min, x_max, y_min, y_max] for imshow
    """
    # Compute canvas dimensions
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    cell_size = canvas_scale
    width = int((x_max - x_min) / cell_size) + 2
    height = int((y_max - y_min) / cell_size) + 2
    
    # Initialize canvas
    if agg == 'max':
        canvas = np.full((height, width), np.nan, dtype=np.float32)
        count = np.zeros((height, width), dtype=np.int32)
    else:
        canvas = np.zeros((height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.int32)
    
    # Fill canvas
    for i in range(len(coords)):
        cx = int((coords[i, 0] - x_min) / cell_size)
        cy = int((coords[i, 1] - y_min) / cell_size)
        cx = min(cx, width - 1)
        cy = min(cy, height - 1)
        
        if agg == 'max':
            if np.isnan(canvas[cy, cx]) or attention[i] > canvas[cy, cx]:
                canvas[cy, cx] = attention[i]
        else:
            canvas[cy, cx] += attention[i]
        count[cy, cx] += 1
    
    # Average for mean aggregation
    if agg == 'mean':
        mask = count > 0
        canvas[mask] = canvas[mask] / count[mask]
        canvas[~mask] = np.nan
    
    extent = [x_min, x_max, y_max, y_min]  # Note: y inverted for image convention
    
    return canvas, extent


def smooth_canvas(canvas, sigma=1.0):
    """Apply Gaussian smoothing to canvas."""
    try:
        from scipy.ndimage import gaussian_filter
        # Replace NaN with 0 for smoothing, then restore
        nan_mask = np.isnan(canvas)
        canvas_filled = np.where(nan_mask, 0, canvas)
        smoothed = gaussian_filter(canvas_filled, sigma=sigma)
        smoothed[nan_mask] = np.nan
        return smoothed
    except ImportError:
        warnings.warn("[WARNING] scipy not available, skipping smoothing")
        return canvas


# ============================================================
# Tissue outline computation
# ============================================================

def compute_tissue_outline(coords, canvas_scale=32):
    """Generate tissue occupancy mask from coords and extract contour.
    
    Args:
        coords: np.ndarray [N, 2], patch coordinates
        canvas_scale: scale factor for canvas
    
    Returns:
        contours: list of contour arrays (each [K, 2] in canvas coords)
        extent: [x_min, x_max, y_min, y_max]
        canvas_shape: (H, W)
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    cell_size = canvas_scale
    width = int((x_max - x_min) / cell_size) + 2
    height = int((y_max - y_min) / cell_size) + 2
    
    # Create binary mask
    mask = np.zeros((height, width), dtype=bool)
    for i in range(len(coords)):
        cx = int((coords[i, 0] - x_min) / cell_size)
        cy = int((coords[i, 1] - y_min) / cell_size)
        cx = min(cx, width - 1)
        cy = min(cy, height - 1)
        mask[cy, cx] = True
    
    # Fill holes
    try:
        from scipy.ndimage import binary_fill_holes
        mask = binary_fill_holes(mask)
    except ImportError:
        pass
    
    # Extract contours
    contours = []
    try:
        from skimage.measure import find_contours
        raw_contours = find_contours(mask.astype(float), 0.5)
        for c in raw_contours:
            # Convert from canvas coords to original coords
            c_orig = np.zeros_like(c)
            c_orig[:, 0] = c[:, 1] * cell_size + x_min
            c_orig[:, 1] = c[:, 0] * cell_size + y_min
            contours.append(c_orig)
    except ImportError:
        # Fallback: use boundary pixels
        try:
            from scipy.ndimage import binary_dilation
            boundary = mask & ~binary_dilation(mask, iterations=1)
            ys, xs = np.where(boundary)
            if len(xs) > 0:
                contour = np.column_stack([xs * cell_size + x_min, ys * cell_size + y_min])
                contours.append(contour)
        except ImportError:
            pass
    
    extent = [x_min, x_max, y_max, y_min]
    return contours, extent, (height, width)


def get_topk_supernodes(supernode_df, top_k=10):
    """Get top-k supernodes by attention.
    
    Args:
        supernode_df: DataFrame with columns ['attention', 'x_min', 'x_max', 'y_min', 'y_max']
        top_k: number of top supernodes to return
    
    Returns:
        list of dicts with supernode info
    """
    df = supernode_df.nlargest(top_k, 'attention')
    results = []
    for _, row in df.iterrows():
        results.append({
            'supernode_id': int(row.get('supernode_id', 0)),
            'attention': float(row['attention']),
            'x_min': float(row['x_min']),
            'x_max': float(row['x_max']),
            'y_min': float(row['y_min']),
            'y_max': float(row['y_max']),
        })
    return results


# ============================================================
# Panel figure generation
# ============================================================

def generate_attention_panel_figure(
    coords,
    patch_attention,
    supernode_df=None,
    slide_id='slide',
    output_path=None,
    background_image_path=None,
    label=None,
    pred=None,
    prob=None,
    cmap_style='soft_pathology',
    clip_percentile=99,
    low_clip_percentile=1,
    gamma=0.7,
    canvas_scale=32,
    smooth_sigma=1.0,
    alpha=0.65,
    dpi=300,
    show_tissue_outline=True,
    show_topk_supernodes=True,
    top_k_supernodes=10,
    outline_color='#31a354',
    outline_width=1.0,
    topk_outline_color='#238b45',
    attention_title='Projected MIL Super-node Attention',
    save_pdf=False,
):
    """Generate publication-quality three-panel attention figure.
    
    Panel A: H&E / tissue background
    Panel B: Tissue outline + top-k supernode bounding boxes
    Panel C: Attention heatmap with colorbar
    
    Returns:
        output_path: path to saved PNG
    """
    # Normalize attention
    attn_norm = normalize_attention(patch_attention, low_clip=low_clip_percentile,
                                     high_clip=clip_percentile, gamma=gamma)
    
    # Rasterize for heatmap panel
    canvas, extent = rasterize_attention(coords, attn_norm, canvas_scale=canvas_scale, agg='max')
    if smooth_sigma > 0:
        canvas = smooth_canvas(canvas, sigma=smooth_sigma)
    
    # Compute tissue outline
    contours, _, _ = compute_tissue_outline(coords, canvas_scale=canvas_scale)
    
    # Get top supernodes
    top_sn = []
    if supernode_df is not None and show_topk_supernodes:
        top_sn = get_topk_supernodes(supernode_df, top_k=top_k_supernodes)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=dpi)
    for ax in axes:
        ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # --- Panel A: Tissue map ---
    ax_a = axes[0]
    if background_image_path and os.path.exists(background_image_path):
        try:
            import cv2
            img = cv2.imread(background_image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax_a.imshow(img, extent=extent, aspect='auto', alpha=0.9)
        except Exception:
            # Fallback to tissue scatter
            ax_a.scatter(coords[:, 0], coords[:, 1], c='#d9d9d9', s=0.5,
                        edgecolors='none', rasterized=True)
    else:
        # Tissue scatter as placeholder
        ax_a.scatter(coords[:, 0], coords[:, 1], c='#d9d9d9', s=0.5,
                    edgecolors='none', rasterized=True)
    
    ax_a.set_title('Tissue map', fontsize=9, pad=6)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.invert_yaxis()
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    
    # --- Panel B: Outline + top supernodes ---
    ax_b = axes[1]
    ax_b.scatter(coords[:, 0], coords[:, 1], c='#e8e8e8', s=0.3,
                edgecolors='none', rasterized=True)
    
    # Draw tissue outline
    if show_tissue_outline and contours:
        for contour in contours:
            ax_b.plot(contour[:, 0], contour[:, 1], color=outline_color,
                     linewidth=outline_width, alpha=0.8)
    
    # Draw top supernode boxes
    for sn in top_sn:
        rect = Rectangle(
            (sn['x_min'], sn['y_min']),
            sn['x_max'] - sn['x_min'],
            sn['y_max'] - sn['y_min'],
            linewidth=0.8,
            edgecolor=topk_outline_color,
            facecolor='none',
            alpha=0.7,
        )
        ax_b.add_patch(rect)
    
    ax_b.set_title('Top-attended super-nodes', fontsize=9, pad=6)
    ax_b.set_aspect('equal', adjustable='box')
    ax_b.invert_yaxis()
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    
    # --- Panel C: Attention heatmap ---
    ax_c = axes[1 + 1]  # axes[2]
    cmap = get_colormap(cmap_style)
    
    im = ax_c.imshow(canvas, cmap=cmap, extent=extent, aspect='auto',
                     vmin=0, vmax=1, alpha=alpha, interpolation='bilinear')
    
    # Draw tissue outline on heatmap too
    if show_tissue_outline and contours:
        for contour in contours:
            ax_c.plot(contour[:, 0], contour[:, 1], color='white',
                     linewidth=0.5, alpha=0.6)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_c, shrink=0.7, pad=0.02, ticks=[0, 0.5, 1.0])
    cbar.set_label('Attention', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    ax_c.set_title(attention_title, fontsize=9, pad=6)
    ax_c.set_aspect('equal', adjustable='box')
    ax_c.invert_yaxis()
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    
    # Add prediction info as subtitle
    pred_str = ''
    if pred is not None:
        pred_str = f'Pred={pred}'
    if label is not None:
        pred_str += f'  Label={label}'
    if prob is not None:
        prob_val = prob[1] if len(prob) > 1 else prob[0]
        pred_str += f'  P(LUSC)={prob_val:.3f}'
    
    if pred_str:
        fig.suptitle(f'{slide_id}  {pred_str}', fontsize=8, y=0.98, color='#555555')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        
        if save_pdf:
            pdf_path = output_path.replace('.png', '.pdf')
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05, facecolor='white')
    
    plt.close(fig)
    return output_path


# ============================================================
# Alignment verification
# ============================================================

def verify_alignment(coords, hilbert_idx, raw_feature_path, hilbert_feature_path,
                     n_samples=20, seed=42):
    """Verify that raw_feature[hilbert_idx] == hilbert_feature."""
    report = {
        'status': 'PASS',
        'checks': {},
        'errors': [],
        'warnings': [],
        'sampled_tokens': [],
    }
    
    if raw_feature_path is None or not os.path.exists(raw_feature_path):
        report['status'] = 'SKIP'
        report['warnings'].append('Raw feature file not provided or not found')
        return report
    
    raw_feat = torch.load(raw_feature_path, map_location='cpu')
    if isinstance(raw_feat, torch.Tensor):
        raw_feat = raw_feat.numpy()
    
    hilbert_feat = torch.load(hilbert_feature_path, map_location='cpu')
    if isinstance(hilbert_feat, torch.Tensor):
        hilbert_feat = hilbert_feat.numpy()
    
    N = len(coords)
    
    len_ok = (len(hilbert_idx) == N == raw_feat.shape[0] == hilbert_feat.shape[0])
    report['checks']['length_consistency'] = {'passed': len_ok}
    if not len_ok:
        report['status'] = 'FAIL'
        report['errors'].append('Length mismatch')
        return report
    
    is_perm = (np.sort(hilbert_idx) == np.arange(N)).all()
    report['checks']['hilbert_idx_is_permutation'] = {'passed': bool(is_perm)}
    if not is_perm:
        report['status'] = 'FAIL'
        report['errors'].append('hilbert_idx is not a valid permutation')
        return report
    
    reordered_raw = raw_feat[hilbert_idx]
    allclose = np.allclose(reordered_raw, hilbert_feat, atol=1e-5)
    max_diff = float(np.abs(reordered_raw - hilbert_feat).max())
    report['checks']['feature_alignment'] = {
        'passed': bool(allclose),
        'max_abs_diff': max_diff,
    }
    if not allclose:
        report['status'] = 'FAIL'
        report['errors'].append(f'Feature alignment failed (max_diff={max_diff:.6f})')
    
    coord_has_nan = np.isnan(coords).any()
    coord_has_inf = np.isinf(coords).any()
    report['checks']['coord_sanity'] = {
        'passed': not (coord_has_nan or coord_has_inf),
    }
    
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(N, size=min(n_samples, N), replace=False)
    sample_indices.sort()
    
    for idx in sample_indices:
        orig_idx = hilbert_idx[idx]
        token_info = {
            'token_idx': int(idx),
            'raw_idx': int(orig_idx),
            'x': float(coords[idx, 0]),
            'y': float(coords[idx, 1]),
            'feature_match': bool(np.allclose(hilbert_feat[idx], reordered_raw[idx], atol=1e-5)),
        }
        report['sampled_tokens'].append(token_info)
    
    return report


# ============================================================
# Tissue mask
# ============================================================

def load_tissue_mask(mask_path, coords):
    """Load tissue mask and determine which patches are tissue."""
    if mask_path.endswith('.h5'):
        with h5py.File(mask_path, 'r') as f:
            mask = f['tissue_mask'][()] if 'tissue_mask' in f else f['mask'][()]
    elif mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        raise ValueError(f"Unsupported mask format: {mask_path}")
    
    tissue_flags = []
    for i in range(len(coords)):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            tissue_flags.append(mask[y, x] > 0)
        else:
            tissue_flags.append(False)
    return np.array(tissue_flags)


def check_tissue_background(patch_attention, coords, tissue_mask=None):
    """Check if high-attention patches are on tissue."""
    report = {
        'tissue_mask_available': tissue_mask is not None,
        'warning': None,
    }
    
    if tissue_mask is None:
        report['warning'] = 'No tissue mask provided. Heatmap requires manual pathology review.'
        report['high_attn_tissue_ratio'] = None
        return report
    
    attn_threshold = np.percentile(patch_attention, 90)
    high_attn = patch_attention >= attn_threshold
    high_attn_tissue = high_attn & tissue_mask
    
    report['high_attn_tissue_ratio'] = float(high_attn_tissue.sum() / max(high_attn.sum(), 1))
    report['total_tissue_ratio'] = float(tissue_mask.sum() / len(tissue_mask))
    
    return report


# ============================================================
# Core data loading
# ============================================================

def load_coords_from_h5(h5_path):
    """Load patch coordinates from an HDF5 file."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        if 'coords' not in f:
            raise KeyError(f"'coords' not in {h5_path}")
        coords = f['coords'][()]
    return coords


def load_hilbert_index(hilbert_pt_path):
    """Load Hilbert curve ordering index from a .pt file."""
    if not os.path.exists(hilbert_pt_path):
        raise FileNotFoundError(f"Hilbert index file not found: {hilbert_pt_path}")
    idx = torch.load(hilbert_pt_path, map_location='cpu')
    if isinstance(idx, torch.Tensor):
        idx = idx.numpy()
    return idx.astype(int)


def reorder_coords_by_hilbert(coords, hilbert_idx):
    """Reorder coordinates according to Hilbert curve ordering."""
    if len(coords) != len(hilbert_idx):
        raise ValueError(f"Length mismatch: coords={len(coords)}, hilbert_idx={len(hilbert_idx)}")
    return coords[hilbert_idx]


def map_supernode_to_patches(n_patches, pool_size, supernode_idx):
    """Map a super-node index to its corresponding patch indices."""
    start = supernode_idx * pool_size
    end = min((supernode_idx + 1) * pool_size, n_patches)
    return list(range(start, end))


def compute_patch_attention(supernode_attention, n_patches, pool_size, mapping_mode='assign'):
    """Distribute super-node attention weights to individual patches."""
    M = len(supernode_attention)
    patch_attention = np.zeros(n_patches, dtype=np.float64)
    for m in range(M):
        start = m * pool_size
        end = min((m + 1) * pool_size, n_patches)
        if start >= n_patches:
            break
        if mapping_mode == 'distribute':
            patch_attention[start:end] = supernode_attention[m] / (end - start)
        else:
            patch_attention[start:end] = supernode_attention[m]
    return patch_attention


# ============================================================
# Legacy scatter heatmap (kept for backward compatibility)
# ============================================================

def generate_topology_heatmap(coords, patch_attention, slide_id, save_path,
                               title=None, figsize=(10, 10),
                               cmap_style='soft_pathology', clip_percentile=99,
                               low_clip_percentile=1, gamma=0.7,
                               alpha=0.7, point_size=4, dpi=300,
                               hide_axes=False, variant='paper',
                               stitch_image_path=None):
    """Generate a scatter-plot heatmap of patch-level attention.
    
    DEPRECATED for paper use. Use generate_attention_panel_figure() instead.
    Kept for backward compatibility and debug use.
    """
    # Warning for jet
    if cmap_style == 'jet':
        warnings.warn("[WARNING] jet is not recommended for publication heatmaps.")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Use proper colormap
    cmap = get_colormap(cmap_style)
    
    # Proper normalization with clipping and gamma
    attn_norm = normalize_attention(patch_attention, low_clip=low_clip_percentile,
                                     high_clip=clip_percentile, gamma=gamma)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=attn_norm, cmap=cmap, s=point_size, alpha=alpha,
        edgecolors='none', rasterized=True,
        vmin=0, vmax=1
    )
    
    if variant == 'debug':
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_label('Projected MIL Super-node Attention')
        if title is None:
            title = f'DEBUG projected super-node attention: {slide_id}'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
    else:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, ticks=[0, 0.5, 1.0])
        cbar.set_label('Attention', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        if title:
            ax.set_title(title, fontsize=9, pad=8)
        if hide_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)


# ============================================================
# Full pipeline
# ============================================================

def process_slide_heatmap(slide_id, coords_h5_path, hilbert_pt_path,
                           feature_path, model, pool_size=50,
                           output_dir=None, device=None,
                           attention_mapping='assign',
                           pred=None, prob=None, label=None,
                           raw_feature_path=None,
                           tissue_mask_path=None,
                           stitch_image_path=None,
                           background_image_path=None,
                           cmap_style='soft_pathology',
                           clip_percentile=99,
                           low_clip_percentile=1,
                           gamma=0.7,
                           smooth_sigma=1.0,
                           canvas_scale=32,
                           alpha=0.65,
                           point_size=1.5,
                           dpi=300,
                           hide_axes=True,
                           vis_mode='panel',
                           show_tissue_outline=True,
                           show_topk_supernodes=True,
                           top_k_supernodes=10,
                           outline_color='#31a354',
                           topk_outline_color='#238b45',
                           force_paper_heatmap=False,
                           save_pdf=True):
    """Full pipeline: load data, verify alignment, run model, compute attention, generate heatmaps.
    
    Args:
        vis_mode: 'scatter', 'raster', or 'panel'
        Other args: see CLI help
    
    Returns:
        dict with alignment report, attention data, and output paths
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data
    coords = load_coords_from_h5(coords_h5_path)
    hilbert_idx = load_hilbert_index(hilbert_pt_path)
    
    if len(hilbert_idx) != len(coords):
        raise ValueError(f"Hilbert index length ({len(hilbert_idx)}) != coords length ({len(coords)}).")
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    features = torch.load(feature_path, map_location='cpu')
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    reordered_coords = reorder_coords_by_hilbert(coords, hilbert_idx)
    
    if features.shape[0] != reordered_coords.shape[0]:
        raise ValueError(f"Feature count ({features.shape[0]}) != coord count ({reordered_coords.shape[0]}).")
    
    N = features.shape[0]
    expected_M = (N + pool_size - 1) // pool_size
    
    # 2. Alignment verification
    alignment_report = verify_alignment(coords, hilbert_idx, raw_feature_path, feature_path)
    
    if output_dir is None:
        output_dir = os.path.join('heatmap_output', slide_id)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'alignment_report.json'), 'w') as f:
        json.dump(alignment_report, f, indent=2, default=str)
    
    print(f"  Alignment status: {alignment_report['status']}")
    
    # 3. Load tissue mask
    tissue_mask = None
    if tissue_mask_path and os.path.exists(tissue_mask_path):
        try:
            tissue_mask = load_tissue_mask(tissue_mask_path, reordered_coords)
            print(f"  Tissue mask: {tissue_mask.sum()}/{len(tissue_mask)} tissue patches")
        except Exception as e:
            print(f"  [WARN] Could not load tissue mask: {e}")
    
    # 4. Run model
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, Y_prob, Y_hat, A_raw, _ = model(features_tensor)
    
    if A_raw is None:
        raise ValueError("Model returned A_raw=None.")
    
    supernode_attn = A_raw.squeeze().cpu().numpy()
    M = len(supernode_attn)
    
    if pred is None:
        pred = Y_hat.item()
    if prob is None:
        prob = Y_prob.squeeze().cpu().numpy()
    
    # 5. Compute patch attention
    patch_attn = compute_patch_attention(supernode_attn, N, pool_size, mapping_mode=attention_mapping)
    attn_norm = normalize_attention(patch_attn, low_clip=low_clip_percentile,
                                     high_clip=clip_percentile, gamma=gamma)
    
    # 6. Tissue/background check
    tissue_report = check_tissue_background(patch_attn, reordered_coords, tissue_mask)
    if tissue_report.get('warning'):
        print(f"  [WARNING] {tissue_report['warning']}")
    
    # 7. Save CSV files
    
    # patch_attention.csv
    sn_ids = []
    for m in range(M):
        start = m * pool_size
        end = min((m + 1) * pool_size, N)
        sn_ids.extend([m] * (end - start))
    sn_ids = sn_ids[:N]
    
    patch_df = pd.DataFrame({
        'slide_id': [slide_id] * N,
        'x': reordered_coords[:, 0],
        'y': reordered_coords[:, 1],
        'token_idx': list(range(N)),
        'supernode_id': sn_ids,
        'attention': patch_attn,
        'attention_norm': attn_norm,
        'pred': [pred] * N,
        'prob_0': [float(prob[0])] * N,
        'prob_1': [float(prob[1])] * N if len(prob) > 1 else [float('nan')] * N,
        'attention_source': ['MIL super-node attention'] * N,
        'projection_method': [attention_mapping] * N,
    })
    if label is not None:
        patch_df['label'] = [label] * N
    if tissue_mask is not None:
        patch_df['is_tissue'] = tissue_mask
    patch_csv_path = os.path.join(output_dir, 'patch_attention.csv')
    patch_df.to_csv(patch_csv_path, index=False)
    
    # supernode_attention.csv
    sn_data = {
        'supernode_id': list(range(M)),
        'attention': supernode_attn.tolist(),
        'n_patches': [],
        'x_mean': [], 'y_mean': [],
        'x_min': [], 'x_max': [], 'y_min': [], 'y_max': [],
    }
    for m in range(M):
        start = m * pool_size
        end = min((m + 1) * pool_size, N)
        window_coords = reordered_coords[start:end]
        sn_data['n_patches'].append(end - start)
        sn_data['x_mean'].append(float(window_coords[:, 0].mean()))
        sn_data['y_mean'].append(float(window_coords[:, 1].mean()))
        sn_data['x_min'].append(float(window_coords[:, 0].min()))
        sn_data['x_max'].append(float(window_coords[:, 0].max()))
        sn_data['y_min'].append(float(window_coords[:, 1].min()))
        sn_data['y_max'].append(float(window_coords[:, 1].max()))
    
    sn_df = pd.DataFrame(sn_data)
    sn_csv_path = os.path.join(output_dir, 'supernode_attention.csv')
    sn_df.to_csv(sn_csv_path, index=False)
    
    # 8. Generate visualizations
    
    output_paths = {
        'alignment_report': os.path.join(output_dir, 'alignment_report.json'),
        'patch_attention_csv': patch_csv_path,
        'supernode_attention_csv': sn_csv_path,
    }
    
    # Debug scatter (always)
    debug_path = os.path.join(output_dir, 'topology_heatmap_scatter_debug.png')
    generate_topology_heatmap(
        reordered_coords, patch_attn, slide_id, debug_path,
        title=f'DEBUG projected super-node attention: {slide_id}',
        cmap_style='jet_debug', clip_percentile=clip_percentile,
        low_clip_percentile=low_clip_percentile, gamma=gamma,
        alpha=alpha, point_size=point_size, dpi=dpi,
        hide_axes=False, variant='debug'
    )
    output_paths['debug_heatmap'] = debug_path
    
    # Paper figures (only if alignment OK)
    can_generate_paper = alignment_report['status'] in ('PASS', 'SKIP') or force_paper_heatmap
    
    if can_generate_paper:
        # Panel figure
        panel_path = os.path.join(output_dir, 'topology_attention_panel.png')
        generate_attention_panel_figure(
            coords=reordered_coords,
            patch_attention=patch_attn,
            supernode_df=sn_df,
            slide_id=slide_id,
            output_path=panel_path,
            background_image_path=background_image_path,
            label=label, pred=pred, prob=prob,
            cmap_style=cmap_style,
            clip_percentile=clip_percentile,
            low_clip_percentile=low_clip_percentile,
            gamma=gamma,
            canvas_scale=canvas_scale,
            smooth_sigma=smooth_sigma,
            alpha=alpha,
            dpi=dpi,
            show_tissue_outline=show_tissue_outline,
            show_topk_supernodes=show_topk_supernodes,
            top_k_supernodes=top_k_supernodes,
            outline_color=outline_color,
            topk_outline_color=topk_outline_color,
            save_pdf=save_pdf,
        )
        output_paths['panel_figure'] = panel_path
        
        # Paper heatmap (scatter style)
        paper_path = os.path.join(output_dir, 'topology_attention_heatmap_paper.png')
        label_str = f'L={label}' if label is not None else ''
        pred_str = f'P={pred}'
        title_paper = f'{slide_id} | {pred_str} {label_str}'.strip(' |')
        generate_topology_heatmap(
            reordered_coords, patch_attn, slide_id, paper_path,
            title=title_paper, cmap_style=cmap_style,
            clip_percentile=clip_percentile,
            low_clip_percentile=low_clip_percentile, gamma=gamma,
            alpha=alpha, point_size=point_size, dpi=dpi,
            hide_axes=hide_axes, variant='paper'
        )
        output_paths['paper_heatmap'] = paper_path
        
        print(f"  Panel figure: {panel_path}")
    else:
        print(f"  [BLOCKED] Paper figures NOT generated (alignment failed)")
    
    # 9. prediction.json
    pred_info = {
        'slide_id': slide_id,
        'pred': int(pred),
        'probabilities': prob.tolist() if hasattr(prob, 'tolist') else list(prob),
        'attention_mapping': attention_mapping,
        'attention_source': 'MIL super-node attention',
        'attention_interpretation': 'class-agnostic aggregation weight, not tumor probability',
        'pool_size': pool_size,
        'n_patches': N,
        'n_supernodes': M,
        'alignment_status': alignment_report['status'],
        'tissue_report': tissue_report,
        'visualization': {
            'cmap_style': cmap_style,
            'vis_mode': vis_mode,
            'clip_percentile': clip_percentile,
            'low_clip_percentile': low_clip_percentile,
            'gamma': gamma,
        },
        'output_files': output_paths,
    }
    if label is not None:
        pred_info['label'] = int(label)
    
    with open(os.path.join(output_dir, 'prediction.json'), 'w') as f:
        json.dump(pred_info, f, indent=2, default=str)
    
    return {
        'alignment_report': alignment_report,
        'patch_attention': patch_attn,
        'supernode_attention': supernode_attn,
        'patch_csv_path': patch_csv_path,
        'supernode_csv_path': sn_csv_path,
        'debug_heatmap_path': debug_path,
        'paper_heatmap_path': output_paths.get('paper_heatmap'),
        'panel_figure_path': output_paths.get('panel_figure'),
        'tissue_report': tissue_report,
    }
