#!/usr/bin/env python3
"""
Spatial Order Diagnostics Tool for IHG-Mamba.

Analyzes spatial continuity of different ordering/sampling strategies:
- raw/raster order
- Hilbert curve order
- Random permutation
- Various sampling modes (random_points, uniform_points, chunk)

Outputs per-slide and aggregate diagnostics CSV + optional plots.

Usage:
    python tools/analyze_spatial_order.py --h5_dir ... --hilbert_idx_dir ... --order_mode hilbert
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_coords(h5_path):
    """Load coords from h5 file."""
    with h5py.File(h5_path, 'r') as f:
        if 'coords' in f:
            return f['coords'][()]
        elif 'coordinates' in f:
            return f['coordinates'][()]
        else:
            raise KeyError(f"No coords in {h5_path}. Keys: {list(f.keys())}")


def load_hilbert_idx(pt_path):
    """Load Hilbert index from .pt file."""
    return torch.load(pt_path).numpy()


def apply_order(coords, order_mode, hilbert_idx=None, seed=1):
    """Apply ordering to coordinates."""
    if order_mode == 'raw':
        return coords.copy()
    elif order_mode == 'hilbert':
        if hilbert_idx is None:
            raise ValueError("hilbert_idx required for hilbert order")
        return coords[hilbert_idx]
    elif order_mode == 'random_perm':
        rng = np.random.RandomState(seed)
        idx = np.arange(len(coords))
        rng.shuffle(idx)
        return coords[idx]
    else:
        raise ValueError(f"Unknown order_mode: {order_mode}")


def apply_sampling(coords, sampling_mode, max_seq_len=2500, chunk_size=50, seed=1):
    """Apply sampling to ordered coordinates. Returns (sampled_coords, sampled_indices)."""
    N = len(coords)
    if sampling_mode == 'none' or max_seq_len <= 0 or max_seq_len >= N:
        return coords.copy(), np.arange(N)
    
    if sampling_mode == 'random_points':
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=max_seq_len, replace=False)
        idx.sort()
        return coords[idx], idx
    elif sampling_mode == 'uniform_points':
        idx = np.linspace(0, N - 1, max_seq_len, dtype=int)
        return coords[idx], idx
    elif sampling_mode == 'chunk':
        # Hilbert chunk sampling
        num_chunks = max_seq_len // chunk_size
        if num_chunks == 0:
            num_chunks = 1
        seg_len = N // num_chunks
        rng = np.random.RandomState(seed)
        indices = []
        for c in range(num_chunks):
            start = c * seg_len
            end = min(start + seg_len, N)
            if end - start <= chunk_size:
                indices.extend(range(start, end))
            else:
                pos = rng.randint(start, end - chunk_size)
                indices.extend(range(pos, pos + chunk_size))
        indices = np.array(indices[:max_seq_len])
        return coords[indices], indices
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")


def compute_jump_distances(coords):
    """Compute adjacent-pair Euclidean distances."""
    if len(coords) < 2:
        return np.array([0.0])
    diffs = np.diff(coords, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    return dists


def compute_tear_rate(dists, threshold):
    """Fraction of adjacent pairs with distance > threshold."""
    return float(np.mean(dists > threshold))


def compute_pool_window_diameters(coords, pool_size):
    """Compute bounding-box diagonal for each pool window."""
    N = len(coords)
    n_windows = (N + pool_size - 1) // pool_size
    diameters = []
    for w in range(n_windows):
        start = w * pool_size
        end = min(start + pool_size, N)
        window = coords[start:end]
        if len(window) < 2:
            diameters.append(0.0)
            continue
        x_min, y_min = window.min(axis=0)
        x_max, y_max = window.max(axis=0)
        diag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        diameters.append(float(diag))
    return np.array(diameters)


def compute_coverage_ratio(coords_full, coords_sampled, grid_size=32):
    """Compute fraction of grid cells covered by sampled coords vs full coords."""
    def grid_cells(coords, grid_size):
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1
        gx = np.clip(((coords[:, 0] - x_min) / (x_max - x_min) * grid_size).astype(int), 0, grid_size - 1)
        gy = np.clip(((coords[:, 1] - y_min) / (y_max - y_min) * grid_size).astype(int), 0, grid_size - 1)
        return set(zip(gx, gy))
    
    full_cells = grid_cells(coords_full, grid_size)
    sampled_cells = grid_cells(coords_sampled, grid_size)
    if len(full_cells) == 0:
        return 0.0
    return len(sampled_cells) / len(full_cells)


def analyze_slide(slide_id, coords_raw, hilbert_idx=None,
                  order_mode='hilbert', sampling_mode='none',
                  max_seq_len=2500, chunk_size=50, pool_size=50,
                  patch_size=512, tear_threshold_factor=3.0,
                  grid_size=32, seed=1):
    """Run full spatial diagnostics for one slide."""
    # Apply ordering
    coords_ordered = apply_order(coords_raw, order_mode, hilbert_idx, seed)
    
    # Apply sampling
    coords_sampled, sampled_idx = apply_sampling(
        coords_ordered, sampling_mode, max_seq_len, chunk_size, seed
    )
    
    N_full = len(coords_ordered)
    N_sampled = len(coords_sampled)
    
    # Jump distances
    jump_dists = compute_jump_distances(coords_sampled)
    tear_threshold = patch_size * tear_threshold_factor
    tear_rate = compute_tear_rate(jump_dists, tear_threshold)
    
    # Pool window diameters
    window_diams = compute_pool_window_diameters(coords_sampled, pool_size)
    
    # Coverage
    coverage = compute_coverage_ratio(coords_raw, coords_sampled, grid_size)
    
    result = {
        'slide_id': slide_id,
        'order_mode': order_mode,
        'sampling_mode': sampling_mode,
        'max_seq_len': max_seq_len,
        'chunk_size': chunk_size,
        'pool_size': pool_size,
        'n_tokens_full': N_full,
        'n_tokens_sampled': N_sampled,
        'mean_jump': float(np.mean(jump_dists)),
        'median_jump': float(np.median(jump_dists)),
        'p90_jump': float(np.percentile(jump_dists, 90)),
        'p99_jump': float(np.percentile(jump_dists, 99)),
        'max_jump': float(np.max(jump_dists)),
        'tear_rate': tear_rate,
        'mean_window_diameter': float(np.mean(window_diams)),
        'median_window_diameter': float(np.median(window_diams)),
        'p90_window_diameter': float(np.percentile(window_diams, 90)),
        'p99_window_diameter': float(np.percentile(window_diams, 99)),
        'coverage_ratio': coverage,
    }
    return result, coords_sampled, jump_dists, window_diams


def save_plots(slide_id, coords_raw, coords_sampled, jump_dists, window_diams, output_dir):
    """Save diagnostic plots for a single slide."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Trajectory
    ax = axes[0, 0]
    ax.scatter(coords_raw[:, 0], coords_raw[:, 1], s=1, alpha=0.3, c='grey', label='Full')
    ax.plot(coords_sampled[:, 0], coords_sampled[:, 1], lw=0.3, alpha=0.7, c='blue')
    ax.scatter(coords_sampled[:, 0], coords_sampled[:, 1], s=2, c='red', zorder=5)
    ax.set_title(f'{slide_id} - Ordered Trajectory')
    ax.invert_yaxis()
    ax.legend()
    
    # 2. Jump distance histogram
    ax = axes[0, 1]
    ax.hist(jump_dists, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(jump_dists), color='red', ls='--', label=f'Mean={np.mean(jump_dists):.1f}')
    ax.set_xlabel('Adjacent Jump Distance')
    ax.set_title('Jump Distance Distribution')
    ax.legend()
    
    # 3. Window diameter histogram
    ax = axes[1, 0]
    ax.hist(window_diams, bins=min(50, len(window_diams)), edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(window_diams), color='red', ls='--', label=f'Mean={np.mean(window_diams):.1f}')
    ax.set_xlabel('Pool Window Diameter')
    ax.set_title('Window Diameter Distribution')
    ax.legend()
    
    # 4. Coverage scatter
    ax = axes[1, 1]
    ax.scatter(coords_raw[:, 0], coords_raw[:, 1], s=1, alpha=0.2, c='grey', label='Full')
    ax.scatter(coords_sampled[:, 0], coords_sampled[:, 1], s=2, c='blue', alpha=0.6, label='Sampled')
    ax.set_title('Coverage: Full vs Sampled')
    ax.invert_yaxis()
    ax.legend()
    
    fig.suptitle(f'Spatial Order Diagnostics: {slide_id}', fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(output_dir, f'{slide_id}_spatial_diagnostics.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Spatial Order Diagnostics for IHG-Mamba')
    parser.add_argument('--h5_dir', type=str, required=True, help='Coords h5 directory')
    parser.add_argument('--hilbert_idx_dir', type=str, default=None, help='Hilbert index directory')
    parser.add_argument('--csv_path', type=str, default=None, help='CSV with slide_id column')
    parser.add_argument('--slide_id', type=str, default=None, help='Single slide to analyze')
    parser.add_argument('--order_mode', type=str, default='hilbert', choices=['raw', 'hilbert', 'random_perm'])
    parser.add_argument('--sampling_mode', type=str, default='none', choices=['none', 'random_points', 'uniform_points', 'chunk'])
    parser.add_argument('--max_seq_len', type=int, default=2500)
    parser.add_argument('--chunk_size', type=int, default=50)
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--tear_threshold_factor', type=float, default=3.0)
    parser.add_argument('--grid_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output_csv', type=str, default='experiments/diagnostics/spatial_order_diagnostics.csv')
    parser.add_argument('--save_plots', action='store_true')
    parser.add_argument('--output_dir', type=str, default='experiments/diagnostics/spatial_order')
    args = parser.parse_args()
    
    # Get slide list
    if args.slide_id:
        slide_ids = [args.slide_id]
    elif args.csv_path:
        df = pd.read_csv(args.csv_path)
        slide_ids = df['slide_id'].tolist()
    else:
        h5_files = [f.replace('.h5', '') for f in os.listdir(args.h5_dir) if f.endswith('.h5')]
        slide_ids = sorted(h5_files)
    
    print(f"Analyzing {len(slide_ids)} slides, order={args.order_mode}, sampling={args.sampling_mode}")
    
    results = []
    for i, sid in enumerate(slide_ids):
        h5_path = os.path.join(args.h5_dir, f'{sid}.h5')
        if not os.path.exists(h5_path):
            warnings.warn(f"H5 not found for {sid}, skipping")
            continue
        
        coords_raw = load_coords(h5_path)
        
        hilbert_idx = None
        if args.order_mode == 'hilbert':
            if args.hilbert_idx_dir is None:
                raise FileNotFoundError("--hilbert_idx_dir required for hilbert order")
            pt_path = os.path.join(args.hilbert_idx_dir, f'{sid}_hilbert.pt')
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Hilbert index not found: {pt_path}")
            hilbert_idx = load_hilbert_idx(pt_path)
            if len(hilbert_idx) != len(coords_raw):
                raise ValueError(f"Hilbert idx length ({len(hilbert_idx)}) != coords ({len(coords_raw)}) for {sid}")
        
        result, coords_sampled, jump_dists, window_diams = analyze_slide(
            sid, coords_raw, hilbert_idx,
            args.order_mode, args.sampling_mode,
            args.max_seq_len, args.chunk_size, args.pool_size,
            args.patch_size, args.tear_threshold_factor,
            args.grid_size, args.seed,
        )
        results.append(result)
        
        if args.save_plots:
            save_plots(sid, coords_raw, coords_sampled, jump_dists, window_diams, args.output_dir)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(slide_ids)}")
    
    # Save per-slide CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nPer-slide diagnostics saved to: {args.output_csv}")
    
    # Save summary
    if len(results) > 1:
        summary = {
            'order_mode': args.order_mode,
            'sampling_mode': args.sampling_mode,
            'max_seq_len': args.max_seq_len,
            'n_slides': len(results),
            'mean_of_mean_jump': float(df['mean_jump'].mean()),
            'mean_tear_rate': float(df['tear_rate'].mean()),
            'mean_coverage_ratio': float(df['coverage_ratio'].mean()),
            'mean_window_diameter': float(df['mean_window_diameter'].mean()),
        }
        summary_csv = args.output_csv.replace('.csv', '_summary.csv')
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        print(f"Summary saved to: {summary_csv}")
    
    print("\n=== Summary Statistics ===")
    print(f"  Slides analyzed: {len(results)}")
    print(f"  Mean jump distance: {df['mean_jump'].mean():.2f}")
    print(f"  Mean tear rate: {df['tear_rate'].mean():.4f}")
    print(f"  Mean coverage ratio: {df['coverage_ratio'].mean():.4f}")
    print(f"  Mean window diameter: {df['mean_window_diameter'].mean():.2f}")


if __name__ == '__main__':
    main()
