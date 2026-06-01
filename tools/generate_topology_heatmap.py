#!/usr/bin/env python3
"""
Generate topology-aware attention heatmap for a slide.

Usage:
    python tools/generate_topology_heatmap.py \
        --slide_id TCGA-XX-XXXX-01Z-00-DX1 \
        --coords_h5 /path/to/patches/slide.h5 \
        --hilbert_pt /path/to/hilbert/slide_hilbert.pt \
        --feature_pt /path/to/pt_files/slide.pt \
        --checkpoint /path/to/s_checkpoint.pt \
        --pool_size 50 \
        --output_dir ./heatmap_output

This script:
  1. Loads patch coordinates from the H5 file
  2. Loads Hilbert curve ordering index
  3. Reorders coordinates to match Hilbert-ordered features
  4. Runs the model forward pass to extract super-node attention
  5. Maps super-node attention to individual patches
  6. Outputs: patch_attention.csv, supernode_attention.csv, topology_heatmap_scatter.png
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch

from utils.heatmap_utils import process_slide_heatmap


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build model
    from models.MambaMIL import MambaMIL
    model = MambaMIL(
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        dropout=0.0,
        act='gelu',
        survival=args.survival,
        hidden_dim=args.hidden_dim,
        local_layers=args.local_layers,
        global_layers=args.global_layers,
        pool_size=args.pool_size,
        use_atp_pool=not args.disable_atp_pool,
        pool_mode=args.pool_mode,
        diffusion_steps=args.diffusion_steps,
        K_init=args.K_init,
        attn_type=args.attn_type,
        attn_dim=args.attn_dim,
    )
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    # Process slide
    print(f"Processing slide: {args.slide_id}")
    result = process_slide_heatmap(
        slide_id=args.slide_id,
        coords_h5_path=args.coords_h5,
        hilbert_pt_path=args.hilbert_pt,
        feature_path=args.feature_pt,
        model=model,
        pool_size=args.pool_size,
        output_dir=args.output_dir,
        device=device,
    )
    
    print(f"\nResults:")
    print(f"  Patch attention CSV:     {result['patch_csv_path']}")
    print(f"  Supernode attention CSV: {result['supernode_csv_path']}")
    print(f"  Heatmap image:           {result['heatmap_path']}")
    print(f"  Number of patches:       {len(result['patch_attention'])}")
    print(f"  Number of super-nodes:   {len(result['supernode_attention'])}")
    print(f"  Attention range:         [{result['patch_attention'].min():.6f}, "
          f"{result['patch_attention'].max():.6f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate topology-aware attention heatmap')
    
    # Required
    parser.add_argument('--slide_id', type=str, required=True, help='Slide identifier')
    parser.add_argument('--coords_h5', type=str, required=True, help='Path to H5 file with patch coordinates')
    parser.add_argument('--hilbert_pt', type=str, required=True, help='Path to Hilbert index .pt file')
    parser.add_argument('--feature_pt', type=str, required=True, help='Path to Hilbert-ordered feature .pt file')
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--in_dim', type=int, default=1024, help='Feature input dimension')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--local_layers', type=int, default=1, help='Number of local Mamba layers')
    parser.add_argument('--global_layers', type=int, default=1, help='Number of global Mamba layers')
    parser.add_argument('--pool_size', type=int, default=50, help='ATP-Pool window size')
    parser.add_argument('--disable_atp_pool', action='store_true', help='Disable ATP-Pool')
    parser.add_argument('--pool_mode', type=str, default='avg', choices=['avg', 'diffusion', 'residual'])
    parser.add_argument('--diffusion_steps', type=int, default=0)
    parser.add_argument('--K_init', type=float, default=2.5)
    parser.add_argument('--attn_type', type=str, default='simple', choices=['simple', 'gated'])
    parser.add_argument('--attn_dim', type=int, default=128)
    parser.add_argument('--survival', action='store_true', help='Use survival mode')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: heatmap_output/<slide_id>)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join('heatmap_output', args.slide_id)
    
    main(args)
