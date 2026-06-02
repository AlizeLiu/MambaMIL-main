"""
Heatmap utilities for topology-aware attention visualization.
Maps super-node attention weights to patch-level coordinates using Hilbert curve ordering.
"""
import os
import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_coords_from_h5(h5_path):
    """Load patch coordinates from an HDF5 file.
    
    Args:
        h5_path: path to .h5 file containing 'coords' dataset
    
    Returns:
        coords: np.ndarray of shape [N, 2] (x, y coordinates)
    
    Raises:
        FileNotFoundError: if h5_path does not exist
        KeyError: if 'coords' dataset not in h5 file
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        if 'coords' not in f:
            raise KeyError(f"'coords' dataset not found in {h5_path}. "
                           f"Available keys: {list(f.keys())}")
        coords = f['coords'][()]
    
    return coords


def load_hilbert_index(hilbert_pt_path):
    """Load Hilbert curve ordering index from a .pt file.
    
    Args:
        hilbert_pt_path: path to .pt file containing the Hilbert sort index
    
    Returns:
        hilbert_idx: np.ndarray of shape [N], integer indices mapping 
                     original position -> Hilbert-ordered position
    
    Raises:
        FileNotFoundError: if file does not exist
    """
    if not os.path.exists(hilbert_pt_path):
        raise FileNotFoundError(f"Hilbert index file not found: {hilbert_pt_path}")
    
    idx = torch.load(hilbert_pt_path, map_location='cpu')
    if isinstance(idx, torch.Tensor):
        idx = idx.numpy()
    return idx.astype(int)


def reorder_coords_by_hilbert(coords, hilbert_idx):
    """Reorder coordinates according to Hilbert curve ordering.
    
    Args:
        coords: np.ndarray [N, 2], original patch coordinates
        hilbert_idx: np.ndarray [N], permutation index from Hilbert sorting
    
    Returns:
        reordered_coords: np.ndarray [N, 2], Hilbert-ordered coordinates
    
    Raises:
        ValueError: if lengths mismatch
    """
    if len(coords) != len(hilbert_idx):
        raise ValueError(
            f"Coordinate count ({len(coords)}) does not match "
            f"Hilbert index length ({len(hilbert_idx)}). "
            f"Ensure coords and hilbert_idx correspond to the same slide."
        )
    
    # hilbert_idx[i] = original position of the i-th Hilbert-ordered patch
    # So coords[hilbert_idx[i]] gives the coordinate of the i-th Hilbert patch
    reordered_coords = coords[hilbert_idx]
    return reordered_coords


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
    """Distribute super-node attention weights to individual patches.
    
    Args:
        supernode_attention: np.ndarray [M], attention weight per super-node
        n_patches: total number of patches
        pool_size: pooling window size
        mapping_mode: 'assign' or 'distribute'
            - 'assign': each patch in supernode gets the supernode's attention value
            - 'distribute': supernode attention divided equally among its patches
    
    Returns:
        patch_attention: np.ndarray [N], attention weight per patch
    """
    M = len(supernode_attention)
    patch_attention = np.zeros(n_patches, dtype=np.float64)
    
    for m in range(M):
        start = m * pool_size
        end = min((m + 1) * pool_size, n_patches)
        if start >= n_patches:
            break
        if mapping_mode == 'distribute':
            patch_attention[start:end] = supernode_attention[m] / (end - start)
        else:  # assign (default)
            patch_attention[start:end] = supernode_attention[m]
    
    return patch_attention


def generate_topology_heatmap(coords, patch_attention, slide_id, save_path,
                               title=None, figsize=(10, 10)):
    """Generate a scatter-plot heatmap of patch-level attention.
    
    Args:
        coords: np.ndarray [N, 2], (x, y) coordinates for each patch
        patch_attention: np.ndarray [N], attention weight per patch
        slide_id: slide identifier for title
        save_path: path to save the figure
        title: optional custom title
        figsize: figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=patch_attention, cmap='hot', s=5, alpha=0.8,
        edgecolors='none'
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight')
    
    if title is None:
        title = f'Topology-aware Attention Heatmap: {slide_id}'
    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_slide_heatmap(slide_id, coords_h5_path, hilbert_pt_path,
                           feature_path, model, pool_size=50,
                           output_dir=None, device=None,
                           attention_mapping='assign',
                           pred=None, prob=None, label=None):
    """Full pipeline: load data, run model, compute attention, generate heatmap.
    
    Args:
        slide_id: slide identifier
        coords_h5_path: path to .h5 file with coordinates
        hilbert_pt_path: path to .pt file with Hilbert index
        feature_path: path to .pt file with features
        model: MambaMIL model (eval mode)
        pool_size: pooling window size used in the model
        output_dir: directory for output files
        device: torch device
        attention_mapping: 'assign' or 'distribute'
        pred: predicted class (optional)
        prob: prediction probabilities (optional)
        label: ground truth label (optional)
    
    Returns:
        dict with 'patch_attention', 'supernode_attention', and output paths
    
    Raises:
        FileNotFoundError: if any required file is missing
        ValueError: if feature/coordinate dimensions mismatch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load coordinates
    coords = load_coords_from_h5(coords_h5_path)
    
    # 2. Load Hilbert index
    hilbert_idx = load_hilbert_index(hilbert_pt_path)
    
    # 3. Validate hilbert_idx length
    if len(hilbert_idx) != len(coords):
        raise ValueError(
            f"Hilbert index length ({len(hilbert_idx)}) != coords length ({len(coords)}). "
            f"Ensure hilbert_idx matches the coordinate file."
        )
    
    # 4. Load features
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    features = torch.load(feature_path, map_location='cpu')
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    # 5. Validate: features should already be Hilbert-ordered
    reordered_coords = reorder_coords_by_hilbert(coords, hilbert_idx)
    
    if features.shape[0] != reordered_coords.shape[0]:
        raise ValueError(
            f"Feature count ({features.shape[0]}) does not match "
            f"reordered coordinate count ({reordered_coords.shape[0]}). "
            f"Ensure features are Hilbert-ordered and coords/hilbert_idx match."
        )
    
    N = features.shape[0]
    expected_M = (N + pool_size - 1) // pool_size
    
    # 6. Run model forward pass to get attention
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)  # [1, N, D]
    
    with torch.no_grad():
        _, Y_prob, Y_hat, A_raw, _ = model(features_tensor)
    
    if A_raw is None:
        raise ValueError("Model returned A_raw=None. Check model configuration.")
    
    # A_raw shape: [1, 1, M] where M = ceil(N / pool_size)
    supernode_attn = A_raw.squeeze().cpu().numpy()  # [M]
    M = len(supernode_attn)
    
    if M != expected_M:
        raise ValueError(
            f"A_raw supernode count ({M}) != expected ({expected_M}). "
            f"N={N}, pool_size={pool_size}."
        )
    
    # Get prediction info if not provided
    if pred is None:
        pred = Y_hat.item()
    if prob is None:
        prob = Y_prob.squeeze().cpu().numpy()
    
    # 7. Map super-node attention to patches
    patch_attn = compute_patch_attention(supernode_attn, N, pool_size, mapping_mode=attention_mapping)
    attn_max = patch_attn.max()
    attn_min = patch_attn.min()
    attn_norm = (patch_attn - attn_min) / (attn_max - attn_min + 1e-8)
    
    # 8. Save outputs
    if output_dir is None:
        output_dir = os.path.join('heatmap_output', slide_id)
    os.makedirs(output_dir, exist_ok=True)
    
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
    patch_csv_path = os.path.join(output_dir, 'patch_attention.csv')
    patch_df.to_csv(patch_csv_path, index=False)
    
    # supernode_attention.csv with coordinate statistics
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
    
    # topology_heatmap_scatter.png
    heatmap_path = os.path.join(output_dir, 'topology_heatmap_scatter.png')
    title = f'{slide_id} | pred={pred} | attn_mapping={attention_mapping}'
    generate_topology_heatmap(reordered_coords, attn_norm, slide_id, heatmap_path, title=title)
    
    # prediction.json
    import json
    pred_info = {
        'slide_id': slide_id,
        'pred': int(pred),
        'probabilities': prob.tolist() if hasattr(prob, 'tolist') else list(prob),
        'attention_mapping': attention_mapping,
        'pool_size': pool_size,
        'n_patches': N,
        'n_supernodes': M,
    }
    if label is not None:
        pred_info['label'] = int(label)
    with open(os.path.join(output_dir, 'prediction.json'), 'w') as f:
        json.dump(pred_info, f, indent=2)
    
    return {
        'patch_attention': patch_attn,
        'supernode_attention': supernode_attn,
        'patch_csv_path': patch_csv_path,
        'supernode_csv_path': sn_csv_path,
        'heatmap_path': heatmap_path,
    }
