"""
Heatmap utilities for topology-aware attention visualization.
Maps super-node attention weights to patch-level coordinates using Hilbert curve ordering.

Quality Control:
- Alignment verification: checks raw_feature[hilbert_idx] vs hilbert_feature
- Tissue/background filtering (when mask available)
- Paper-ready colormap styles
"""
import os
import json
import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# Custom colormaps
# ============================================================

def get_pathology_colormap():
    """Pathology-style colormap: deep blue -> cyan -> green -> yellow -> red.
    High contrast, standard heatmap style used in attention visualization."""
    colors = [
        (0.05, 0.05, 0.30),   # deep navy (lowest attention)
        (0.10, 0.20, 0.60),   # blue
        (0.10, 0.50, 0.80),   # cyan-blue
        (0.20, 0.70, 0.60),   # teal
        (0.40, 0.80, 0.30),   # green
        (0.70, 0.90, 0.10),   # yellow-green
        (0.95, 0.85, 0.10),   # yellow
        (0.95, 0.60, 0.10),   # orange
        (0.90, 0.30, 0.10),   # red-orange
        (0.70, 0.10, 0.10),   # dark red (highest attention)
    ]
    return LinearSegmentedColormap.from_list('pathology', colors, N=256)


def get_colormap(style='pathology'):
    """Get colormap by style name.
    
    Args:
        style: 'pathology', 'magma', 'YlOrRd', or any matplotlib cmap name
    
    Returns:
        matplotlib colormap
    """
    if style == 'pathology':
        return get_pathology_colormap()
    elif style == 'magma':
        return plt.cm.magma
    elif style in ('YlOrRd', 'yl_or_rd'):
        return plt.cm.YlOrRd
    else:
        return plt.get_cmap(style)


# ============================================================
# Alignment verification
# ============================================================

def verify_alignment(coords, hilbert_idx, raw_feature_path, hilbert_feature_path,
                     n_samples=20, seed=42):
    """Verify that raw_feature[hilbert_idx] == hilbert_feature.
    
    This ensures:
    1. The hilbert_idx correctly maps raw -> hilbert ordering
    2. The feature file is indeed Hilbert-ordered
    3. coords/hilbert_idx/feature token semantics are consistent
    
    Args:
        coords: np.ndarray [N, 2], raw patch coordinates
        hilbert_idx: np.ndarray [N], permutation index
        raw_feature_path: path to raw (unsorted) feature .pt file, or None
        hilbert_feature_path: path to Hilbert-ordered feature .pt file
        n_samples: number of tokens to sample for detailed report
        seed: random seed for sampling
    
    Returns:
        dict with alignment report
    """
    report = {
        'status': 'PASS',
        'checks': {},
        'errors': [],
        'warnings': [],
        'sampled_tokens': [],
    }
    
    # Skip if raw features not provided
    if raw_feature_path is None or not os.path.exists(raw_feature_path):
        report['status'] = 'SKIP'
        report['warnings'].append('Raw feature file not provided or not found')
        report['warnings'].append('Cannot verify alignment without raw features')
        return report
    
    raw_feat = torch.load(raw_feature_path, map_location='cpu')
    if isinstance(raw_feat, torch.Tensor):
        raw_feat = raw_feat.numpy()
    
    hilbert_feat = torch.load(hilbert_feature_path, map_location='cpu')
    if isinstance(hilbert_feat, torch.Tensor):
        hilbert_feat = hilbert_feat.numpy()
    
    N = len(coords)
    
    # Check 1: Length consistency
    len_ok = (len(hilbert_idx) == N == raw_feat.shape[0] == hilbert_feat.shape[0])
    report['checks']['length_consistency'] = {
        'passed': len_ok,
        'coords': N,
        'hilbert_idx': len(hilbert_idx),
        'raw_features': raw_feat.shape[0],
        'hilbert_features': hilbert_feat.shape[0],
    }
    if not len_ok:
        report['status'] = 'FAIL'
        report['errors'].append('Length mismatch between coords/hilbert_idx/features')
        return report
    
    # Check 2: hilbert_idx is a valid permutation
    is_perm = (np.sort(hilbert_idx) == np.arange(N)).all()
    report['checks']['hilbert_idx_is_permutation'] = {'passed': bool(is_perm)}
    if not is_perm:
        report['status'] = 'FAIL'
        report['errors'].append('hilbert_idx is not a valid permutation')
        return report
    
    # Check 3: raw_feature[hilbert_idx] == hilbert_feature (allclose)
    reordered_raw = raw_feat[hilbert_idx]
    allclose = np.allclose(reordered_raw, hilbert_feat, atol=1e-5)
    max_diff = np.abs(reordered_raw - hilbert_feat).max()
    mean_diff = np.abs(reordered_raw - hilbert_feat).mean()
    
    report['checks']['feature_alignment'] = {
        'passed': bool(allclose),
        'max_abs_diff': float(max_diff),
        'mean_abs_diff': float(mean_diff),
        'atol_used': 1e-5,
    }
    if not allclose:
        report['status'] = 'FAIL'
        report['errors'].append(
            f'raw_feature[hilbert_idx] != hilbert_feature (max_diff={max_diff:.6f})'
        )
    
    # Check 4: Coordinate sanity (no NaN/Inf, reasonable range)
    coord_has_nan = np.isnan(coords).any()
    coord_has_inf = np.isinf(coords).any()
    report['checks']['coord_sanity'] = {
        'passed': not (coord_has_nan or coord_has_inf),
        'has_nan': bool(coord_has_nan),
        'has_inf': bool(coord_has_inf),
        'x_range': [float(coords[:, 0].min()), float(coords[:, 0].max())],
        'y_range': [float(coords[:, 1].min()), float(coords[:, 1].max())],
    }
    if coord_has_nan or coord_has_inf:
        report['status'] = 'FAIL'
        report['errors'].append('Coordinates contain NaN or Inf')
    
    # Sample tokens for detailed inspection
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(N, size=min(n_samples, N), replace=False)
    sample_indices.sort()
    
    for idx in sample_indices:
        orig_idx = hilbert_idx[idx]
        token_info = {
            'token_idx': int(idx),
            'raw_idx': int(orig_idx),
            'x': float(coords[idx, 0]) if idx < len(coords) else None,
            'y': float(coords[idx, 1]) if idx < len(coords) else None,
            'hilbert_feat_norm': float(np.linalg.norm(hilbert_feat[idx])),
            'raw_reordered_norm': float(np.linalg.norm(reordered_raw[idx])),
            'feature_match': bool(np.allclose(hilbert_feat[idx], reordered_raw[idx], atol=1e-5)),
        }
        report['sampled_tokens'].append(token_info)
    
    # Summary
    report['summary'] = {
        'n_tokens': N,
        'n_checked': len(sample_indices),
        'all_passed': report['status'] == 'PASS',
    }
    
    return report


# ============================================================
# Tissue mask filtering
# ============================================================

def load_tissue_mask(mask_path, coords):
    """Load tissue mask and determine which patches are tissue.
    
    Args:
        mask_path: path to tissue mask file (h5 or npy)
        coords: np.ndarray [N, 2], patch coordinates
    
    Returns:
        tissue_mask: np.ndarray [N], boolean (True = tissue)
    """
    if mask_path.endswith('.h5'):
        with h5py.File(mask_path, 'r') as f:
            if 'tissue_mask' in f:
                mask = f['tissue_mask'][()]
            elif 'mask' in f:
                mask = f['mask'][()]
            else:
                raise KeyError(f"No tissue mask found in {mask_path}")
    elif mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        raise ValueError(f"Unsupported mask format: {mask_path}")
    
    # Sample mask at patch coordinates
    tissue_flags = []
    for i in range(len(coords)):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            tissue_flags.append(mask[y, x] > 0)
        else:
            tissue_flags.append(False)
    
    return np.array(tissue_flags)


def check_tissue_background(patch_attention, coords, tissue_mask=None, 
                            tissue_threshold=0.5):
    """Check if high-attention patches are on tissue (not background).
    
    Args:
        patch_attention: np.ndarray [N], attention weights
        coords: np.ndarray [N, 2], coordinates
        tissue_mask: np.ndarray [N], boolean (True = tissue), or None
        tissue_threshold: attention threshold to define "high attention"
    
    Returns:
        dict with tissue/background analysis
    """
    report = {
        'tissue_mask_available': tissue_mask is not None,
        'warning': None,
    }
    
    if tissue_mask is None:
        report['warning'] = 'No tissue mask provided. Manual inspection required.'
        report['high_attn_tissue_ratio'] = None
        return report
    
    # High attention patches
    attn_threshold = np.percentile(patch_attention, 90)
    high_attn = patch_attention >= attn_threshold
    
    # Check what fraction of high-attention patches are on tissue
    high_attn_tissue = high_attn & tissue_mask
    high_attn_background = high_attn & ~tissue_mask
    
    report['attn_threshold_90pctl'] = float(attn_threshold)
    report['n_high_attn'] = int(high_attn.sum())
    report['n_high_attn_on_tissue'] = int(high_attn_tissue.sum())
    report['n_high_attn_on_background'] = int(high_attn_background.sum())
    report['high_attn_tissue_ratio'] = float(high_attn_tissue.sum() / max(high_attn.sum(), 1))
    
    # Overall tissue ratio
    report['total_tissue_ratio'] = float(tissue_mask.sum() / len(tissue_mask))
    
    return report


# ============================================================
# Core heatmap functions (updated)
# ============================================================

def load_coords_from_h5(h5_path):
    """Load patch coordinates from an HDF5 file."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        if 'coords' not in f:
            raise KeyError(f"'coords' dataset not found in {h5_path}. "
                           f"Available keys: {list(f.keys())}")
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
        raise ValueError(
            f"Coordinate count ({len(coords)}) does not match "
            f"Hilbert index length ({len(hilbert_idx)})."
        )
    return coords[hilbert_idx]


def map_supernode_to_patches(n_patches, pool_size, supernode_idx):
    """Map a super-node index to its corresponding patch indices.
    
    After pooling with pool_size, super-node m corresponds to
    patches [m*pool_size, min((m+1)*pool_size, N)).
    
    Args:
        n_patches: total number of patches
        pool_size: pooling window size
        supernode_idx: index of the super-node
    
    Returns:
        list of patch indices
    """
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


def generate_topology_heatmap(coords, patch_attention, slide_id, save_path,
                               title=None, figsize=(10, 10),
                               cmap_style='pathology', clip_percentile=99,
                               alpha=0.7, point_size=4, dpi=300,
                               hide_axes=False, variant='paper',
                               stitch_image_path=None):
    """Generate a scatter-plot heatmap of patch-level attention.
    
    Args:
        coords: np.ndarray [N, 2], (x, y) coordinates for each patch
        patch_attention: np.ndarray [N], attention weight per patch (normalized 0-1)
        slide_id: slide identifier for title
        save_path: path to save the figure
        title: optional custom title
        figsize: figure size
        cmap_style: colormap style ('jet', 'hot', 'viridis', etc.)
        clip_percentile: clip attention values above this percentile (0-100)
        alpha: point transparency
        point_size: scatter point size
        dpi: output resolution
        hide_axes: if True, hide axis labels and ticks
        variant: 'debug' (with axes/title) or 'paper' (clean)
        stitch_image_path: unused, kept for API compatibility
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Use jet colormap: blue -> cyan -> green -> yellow -> red
    cmap = plt.cm.jet
    
    # Normalize attention to 0-1
    attn_min = patch_attention.min()
    attn_max = patch_attention.max()
    if attn_max > attn_min:
        attn_norm = (patch_attention - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(patch_attention)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=attn_norm, cmap=cmap, s=point_size, alpha=alpha,
        edgecolors='none', rasterized=True,
        vmin=0, vmax=1
    )
    
    if variant == 'debug':
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_label('Attention Weight')
        if title is None:
            title = f'DEBUG: {slide_id}'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
    else:
        # Paper style: minimal
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, 
                           ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_label('Attention', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        if title:
            ax.set_title(title, fontsize=9, pad=8)
        if hide_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        else:
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.tick_params(labelsize=7)
    
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # Match image coordinate convention (y down)
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)


def process_slide_heatmap(slide_id, coords_h5_path, hilbert_pt_path,
                           feature_path, model, pool_size=50,
                           output_dir=None, device=None,
                           attention_mapping='assign',
                           pred=None, prob=None, label=None,
                           raw_feature_path=None,
                           tissue_mask_path=None,
                           stitch_image_path=None,
                           cmap_style='pathology',
                           clip_percentile=99,
                           alpha=0.55, point_size=2, dpi=300,
                           hide_axes=False):
    """Full pipeline: load data, verify alignment, run model, compute attention, generate heatmaps.
    
    Args:
        slide_id: slide identifier
        coords_h5_path: path to .h5 file with coordinates
        hilbert_pt_path: path to .pt file with Hilbert index
        feature_path: path to .pt file with Hilbert-ordered features
        model: MambaMIL model (eval mode)
        pool_size: pooling window size used in the model
        output_dir: directory for output files
        device: torch device
        attention_mapping: 'assign' or 'distribute'
        pred: predicted class (optional)
        prob: prediction probabilities (optional)
        label: ground truth label (optional)
        raw_feature_path: path to raw (unsorted) features for alignment check
        tissue_mask_path: path to tissue mask for background filtering
        cmap_style: colormap style for heatmap
        clip_percentile: clip attention above this percentile
        alpha: point transparency
        point_size: scatter point size
        dpi: output DPI
        hide_axes: hide axis labels in paper figure
    
    Returns:
        dict with alignment report, attention data, and output paths
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load coordinates
    coords = load_coords_from_h5(coords_h5_path)
    
    # 2. Load Hilbert index
    hilbert_idx = load_hilbert_index(hilbert_pt_path)
    
    # 3. Validate lengths
    if len(hilbert_idx) != len(coords):
        raise ValueError(
            f"Hilbert index length ({len(hilbert_idx)}) != coords length ({len(coords)})."
        )
    
    # 4. Load features
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    features = torch.load(feature_path, map_location='cpu')
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    # 5. Reorder coords to match Hilbert-ordered features
    reordered_coords = reorder_coords_by_hilbert(coords, hilbert_idx)
    
    if features.shape[0] != reordered_coords.shape[0]:
        raise ValueError(
            f"Feature count ({features.shape[0]}) != reordered coord count ({reordered_coords.shape[0]})."
        )
    
    N = features.shape[0]
    expected_M = (N + pool_size - 1) // pool_size
    
    # 6. Alignment verification
    alignment_report = verify_alignment(
        coords, hilbert_idx, raw_feature_path, feature_path,
        n_samples=20, seed=42
    )
    
    if output_dir is None:
        output_dir = os.path.join('heatmap_output', slide_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save alignment report
    alignment_path = os.path.join(output_dir, 'alignment_report.json')
    with open(alignment_path, 'w') as f:
        json.dump(alignment_report, f, indent=2, default=str)
    
    print(f"  Alignment status: {alignment_report['status']}")
    if alignment_report['warnings']:
        for w in alignment_report['warnings']:
            print(f"    [WARN] {w}")
    if alignment_report['errors']:
        for e in alignment_report['errors']:
            print(f"    [ERROR] {e}")
    
    # 7. Load tissue mask if available
    tissue_mask = None
    if tissue_mask_path and os.path.exists(tissue_mask_path):
        try:
            tissue_mask = load_tissue_mask(tissue_mask_path, reordered_coords)
            print(f"  Tissue mask loaded: {tissue_mask.sum()}/{len(tissue_mask)} tissue patches")
        except Exception as e:
            print(f"  [WARN] Could not load tissue mask: {e}")
    
    # 8. Run model forward pass
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, Y_prob, Y_hat, A_raw, _ = model(features_tensor)
    
    if A_raw is None:
        raise ValueError("Model returned A_raw=None. Check model configuration.")
    
    supernode_attn = A_raw.squeeze().cpu().numpy()
    M = len(supernode_attn)
    
    if M != expected_M:
        raise ValueError(
            f"A_raw supernode count ({M}) != expected ({expected_M})."
        )
    
    # Get prediction info
    if pred is None:
        pred = Y_hat.item()
    if prob is None:
        prob = Y_prob.squeeze().cpu().numpy()
    
    # 9. Map super-node attention to patches
    patch_attn = compute_patch_attention(supernode_attn, N, pool_size, mapping_mode=attention_mapping)
    attn_max = patch_attn.max()
    attn_min = patch_attn.min()
    attn_norm = (patch_attn - attn_min) / (attn_max - attn_min + 1e-8)
    
    # 10. Tissue/background check
    tissue_report = check_tissue_background(patch_attn, reordered_coords, tissue_mask)
    
    # 11. Save CSV files
    
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
    
    # 12. Generate heatmaps
    
    # Always generate debug heatmap
    debug_path = os.path.join(output_dir, 'topology_heatmap_scatter_debug.png')
    title_debug = f'DEBUG {slide_id} | pred={pred} | attn_mapping={attention_mapping}'
    generate_topology_heatmap(
        reordered_coords, attn_norm, slide_id, debug_path,
        title=title_debug, cmap_style=cmap_style, clip_percentile=clip_percentile,
        alpha=alpha, point_size=point_size, dpi=dpi, hide_axes=False, variant='debug',
        stitch_image_path=stitch_image_path
    )
    
    # Paper heatmap: ONLY if alignment passes or is skipped (no raw features)
    paper_path = None
    if alignment_report['status'] in ('PASS', 'SKIP'):
        if alignment_report['status'] == 'SKIP':
            print(f"  [WARN] Alignment check skipped (no raw features). Paper heatmap generated with caveat.")
        paper_path = os.path.join(output_dir, 'topology_heatmap_scatter_paper.png')
        label_str = f'L={label}' if label is not None else ''
        pred_str = f'P={pred}'
        title_paper = f'{slide_id} | {pred_str} {label_str}'.strip(' |')
        generate_topology_heatmap(
            reordered_coords, attn_norm, slide_id, paper_path,
            title=title_paper, cmap_style=cmap_style, clip_percentile=clip_percentile,
            alpha=alpha, point_size=point_size, dpi=dpi, hide_axes=hide_axes, variant='paper',
            stitch_image_path=stitch_image_path
        )
        print(f"  Paper heatmap generated: {paper_path}")
    else:
        print(f"  [BLOCKED] Paper heatmap NOT generated (alignment failed)")
    
    # 13. prediction.json
    pred_info = {
        'slide_id': slide_id,
        'pred': int(pred),
        'probabilities': prob.tolist() if hasattr(prob, 'tolist') else list(prob),
        'attention_mapping': attention_mapping,
        'pool_size': pool_size,
        'n_patches': N,
        'n_supernodes': M,
        'alignment_status': alignment_report['status'],
        'tissue_report': tissue_report,
        'output_files': {
            'alignment_report': alignment_path,
            'debug_heatmap': debug_path,
            'paper_heatmap': paper_path,
            'patch_attention_csv': patch_csv_path,
            'supernode_attention_csv': sn_csv_path,
        },
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
        'paper_heatmap_path': paper_path,
        'tissue_report': tissue_report,
    }
