#!/usr/bin/env python3
"""
Generate topology-aware attention heatmap for a slide.

Usage:
    python tools/generate_topology_heatmap.py \\
        --slide_id TCGA-XX-XXXX-01Z-00-DX1 \\
        --coords_h5 /path/to/patches/slide.h5 \\
        --hilbert_pt /path/to/hilbert/slide_hilbert.pt \\
        --feature_pt /path/to/pt_files/slide.pt \\
        --checkpoint /path/to/s_checkpoint.pt \\
        --pool_size 50 \\
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
    
    # Build model with ALL training parameters
    from models.MambaMIL import MambaMIL
    model = MambaMIL(
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        dropout=0.0,
        act='gelu',
        survival=args.survival,
        layer=1,  # fallback, overridden by local/global_layers
        rate=args.mambamil_rate,
        type=args.mambamil_type,
        hidden_dim=args.hidden_dim,
        local_layers=args.local_layers,
        global_layers=args.global_layers,
        pool_size=args.pool_size,
        use_atp_pool=not args.disable_atp_pool,
        diffusion_steps=args.diffusion_steps,
        K_init=args.K_init,
        atp_dt=args.atp_dt,
        norm_type=args.norm_type,
        pool_mode=args.pool_mode,
        tau_init=args.tau_init,
        gamma_init=args.gamma_init,
        local_segment_mode=args.local_segment_mode,
        local_segment_size=args.local_segment_size,
        attn_type=args.attn_type,
        attn_dim=args.attn_dim,
    )
    
    # Load checkpoint with strict=True
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            print(f"  WARNING: missing keys: {missing}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {unexpected}")
        print("  Checkpoint loaded successfully (strict=True)")
    
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
        attention_mapping=args.attention_mapping,
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
    
    # Model architecture (must match training)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=1)
    parser.add_argument('--global_layers', type=int, default=1)
    parser.add_argument('--mambamil_type', type=str, default='SRMamba', choices=['Mamba', 'BiMamba', 'SRMamba'])
    parser.add_argument('--mambamil_rate', type=int, default=5)
    
    # Pool config
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--disable_atp_pool', action='store_true')
    parser.add_argument('--pool_mode', type=str, default='avg', choices=['avg', 'diffusion', 'residual'])
    parser.add_argument('--diffusion_steps', type=int, default=0)
    parser.add_argument('--K_init', type=float, default=2.5)
    parser.add_argument('--atp_dt', type=float, default=0.1)
    parser.add_argument('--norm_type', type=str, default='mean', choices=['mean', 'sum'])
    parser.add_argument('--tau_init', type=float, default=2.0)
    parser.add_argument('--gamma_init', type=float, default=0.0)
    
    # Segment config
    parser.add_argument('--local_segment_mode', type=str, default='none', choices=['none', 'chunk'])
    parser.add_argument('--local_segment_size', type=int, default=50)
    
    # Attention readout
    parser.add_argument('--attn_type', type=str, default='simple', choices=['simple', 'gated'])
    parser.add_argument('--attn_dim', type=int, default=128)
    
    # Attention mapping
    parser.add_argument('--attention_mapping', type=str, default='assign', choices=['assign', 'distribute'],
                        help='How to map supernode attention to patches: assign (same value) or distribute (divide)')
    
    # Mode
    parser.add_argument('--survival', action='store_true', help='Use survival mode')
    parser.add_argument('--allow_partial_load', action='store_true', help='Allow partial checkpoint loading (strict=False)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: heatmap_output/<slide_id>)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join('heatmap_output', args.slide_id)
    
    main(args)
