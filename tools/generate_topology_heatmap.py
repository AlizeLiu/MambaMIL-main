#!/usr/bin/env python3
"""
Generate topology-aware attention heatmap for a slide.

Quality control features:
  - Alignment verification: checks raw_feature[hilbert_idx] == hilbert_feature
  - Tissue/background filtering (when mask available)
  - Paper-ready colormap styles (pathology, magma, YlOrRd)
  - Debug and paper output variants

Usage:
    python tools/generate_topology_heatmap.py \\
        --slide_id TCGA-XX-XXXX-01Z-00-DX1 \\
        --coords_h5 /path/to/patches/slide.h5 \\
        --hilbert_pt /path/to/hilbert/slide_hilbert.pt \\
        --feature_pt /path/to/hilbert_pt/slide.pt \\
        --checkpoint /path/to/s_checkpoint.pt \\
        --pool_size 50 \\
        --output_dir ./heatmap_output

Outputs:
    alignment_report.json           - alignment verification report
    topology_heatmap_scatter_debug.png  - debug heatmap (with axes)
    topology_heatmap_scatter_paper.png  - paper heatmap (only if alignment passes)
    patch_attention.csv             - per-patch attention values
    supernode_attention.csv         - per-supernode attention values
    prediction.json                 - prediction and QC metadata
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
        layer=1,
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
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        strict = not args.allow_partial_load
        state_dict = torch.load(args.checkpoint, map_location=device)
        
        # Handle backward compatibility for attention key names
        remapped = {}
        needs_remap = False
        for k, v in state_dict.items():
            if k.startswith('attention.0.') or k.startswith('attention.2.'):
                new_key = k.replace('attention.', 'attention.attn.', 1)
                remapped[new_key] = v
                needs_remap = True
            else:
                remapped[k] = v
        if needs_remap:
            print("  [INFO] Remapped old attention keys to SimpleAttention format")
            state_dict = remapped
        
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            print(f"\n[FATAL] Checkpoint loading failed (strict={strict}): {e}")
            if not args.allow_partial_load:
                print("Hint: use --allow_partial_load to force loading (debug only)")
            sys.exit(1)
        if args.allow_partial_load:
            print("  [WARNING] Partial checkpoint loading enabled (debug only)")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print(f"  Checkpoint loaded (strict={strict})")
    
    model = model.to(device)
    model.eval()
    
    # Process slide
    print(f"Processing slide: {args.slide_id}")
    print(f"  Raw feature path: {args.raw_feature_pt or 'NOT PROVIDED (alignment check skipped)'}")
    print(f"  Tissue mask path: {args.tissue_mask or 'NOT PROVIDED (tissue check skipped)'}")
    print(f"  Stitch dir: {args.stitch_dir or 'NOT PROVIDED (no background overlay)'}")
    print(f"  Colormap: {args.cmap_style}, clip_pctl={args.clip_percentile}")
    print(f"  Point: alpha={args.alpha}, size={args.point_size}, dpi={args.dpi}")
    
    # Find stitch image if stitch_dir provided
    stitch_image_path = None
    if args.stitch_dir:
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(args.stitch_dir, f"{args.slide_id}{ext}")
            if os.path.exists(candidate):
                stitch_image_path = candidate
                print(f"  Stitch image: {stitch_image_path}")
                break
        if stitch_image_path is None:
            print(f"  [WARN] No stitch image found for {args.slide_id} in {args.stitch_dir}")
    
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
        raw_feature_path=args.raw_feature_pt,
        tissue_mask_path=args.tissue_mask,
        stitch_image_path=stitch_image_path,
        cmap_style=args.cmap_style,
        clip_percentile=args.clip_percentile,
        alpha=args.alpha,
        point_size=args.point_size,
        dpi=args.dpi,
        hide_axes=args.hide_axes,
    )
    
    # Print summary
    alignment = result['alignment_report']
    print(f"\n{'='*60}")
    print(f"  ALIGNMENT REPORT: {alignment['status']}")
    print(f"{'='*60}")
    for check_name, check_data in alignment.get('checks', {}).items():
        status = "PASS" if check_data.get('passed') else "FAIL"
        print(f"  [{status}] {check_name}")
    
    if alignment.get('sampled_tokens'):
        print(f"\n  Sampled tokens ({len(alignment['sampled_tokens'])}):")
        print(f"  {'token_idx':>10} {'raw_idx':>10} {'x':>8} {'y':>8} {'sn_id':>6} {'match':>6}")
        for t in alignment['sampled_tokens'][:5]:
            print(f"  {t['token_idx']:>10} {t['raw_idx']:>10} "
                  f"{t['x']:>8.0f} {t['y']:>8.0f} "
                  f"{'':>6} {str(t['feature_match']):>6}")
        print(f"  ... ({len(alignment['sampled_tokens'])-5} more)")
    
    print(f"\n  Outputs:")
    print(f"    Alignment report:  {args.output_dir}/alignment_report.json")
    print(f"    Debug heatmap:     {result['debug_heatmap_path']}")
    print(f"    Paper heatmap:     {result['paper_heatmap_path'] or 'BLOCKED (alignment failed)'}")
    print(f"    Patch attention:   {result['patch_csv_path']}")
    print(f"    Supernode attn:    {result['supernode_csv_path']}")
    
    tissue = result['tissue_report']
    if tissue.get('warning'):
        print(f"\n  Tissue check: {tissue['warning']}")
    elif tissue.get('high_attn_tissue_ratio') is not None:
        print(f"\n  Tissue check: {tissue['high_attn_tissue_ratio']:.1%} high-attn patches on tissue")
    
    print(f"\n  Attention range: [{result['patch_attention'].min():.6f}, "
          f"{result['patch_attention'].max():.6f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate topology-aware attention heatmap with QC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument('--slide_id', type=str, required=True, help='Slide identifier')
    parser.add_argument('--coords_h5', type=str, required=True, help='Path to H5 file with patch coordinates')
    parser.add_argument('--hilbert_pt', type=str, required=True, help='Path to Hilbert index .pt file')
    parser.add_argument('--feature_pt', type=str, required=True, help='Path to Hilbert-ordered feature .pt file')
    
    # Alignment verification (optional but recommended)
    parser.add_argument('--raw_feature_pt', type=str, default=None,
                        help='Path to raw (unsorted) feature .pt file for alignment check')
    
    # Tissue mask (optional)
    parser.add_argument('--tissue_mask', type=str, default=None,
                        help='Path to tissue mask file (h5 or npy)')
    
    # Stitch image directory (optional, for background overlay)
    parser.add_argument('--stitch_dir', type=str, default=None,
                        help='Directory containing stitch images (jpg/png) for background overlay')
    
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
                        help='How to map supernode attention to patches')
    
    # Visualization style
    parser.add_argument('--cmap_style', type=str, default='pathology',
                        choices=['pathology', 'magma', 'YlOrRd'],
                        help='Colormap style for heatmap')
    parser.add_argument('--clip_percentile', type=float, default=99,
                        help='Clip attention values above this percentile (0-100)')
    parser.add_argument('--alpha', type=float, default=0.55,
                        help='Point transparency (0-1)')
    parser.add_argument('--point_size', type=float, default=2,
                        help='Scatter point size')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output image DPI')
    parser.add_argument('--hide_axes', action='store_true',
                        help='Hide axis labels and ticks in paper figure')
    
    # Mode
    parser.add_argument('--survival', action='store_true', help='Use survival mode')
    parser.add_argument('--allow_partial_load', action='store_true',
                        help='Allow partial checkpoint loading (debug only)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: heatmap_output/<slide_id>)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join('heatmap_output', args.slide_id)
    
    main(args)
