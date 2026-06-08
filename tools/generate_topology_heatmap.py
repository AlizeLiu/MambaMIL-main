#!/usr/bin/env python3
"""
Generate topology-aware attention heatmap for a slide.

Publication-quality visualization with:
- Three-panel figure (tissue map / outline / attention heatmap)
- Soft pathology colormap
- Gamma-corrected attention normalization
- Tissue outline and top supernode annotations
- Alignment verification

Outputs:
    alignment_report.json                - alignment verification
    topology_attention_panel.png         - publication panel figure
    topology_attention_panel.pdf         - PDF version
    topology_attention_heatmap_paper.png - paper scatter heatmap
    topology_heatmap_scatter_debug.png   - debug scatter (with axes)
    patch_attention.csv                  - per-patch attention
    supernode_attention.csv              - per-supernode attention
    prediction.json                      - prediction and QC metadata
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import warnings

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
            print("  [INFO] Remapped old attention keys")
            state_dict = remapped
        
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            print(f"\n[FATAL] Checkpoint loading failed: {e}")
            sys.exit(1)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print(f"  Checkpoint loaded (strict={strict})")
    
    model = model.to(device)
    model.eval()
    
    # Find background image
    background_image_path = None
    if args.stitch_dir:
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(args.stitch_dir, f"{args.slide_id}{ext}")
            if os.path.exists(candidate):
                background_image_path = candidate
                break
    
    # Print config
    print(f"\nProcessing slide: {args.slide_id}")
    print(f"  Visualization mode: {args.vis_mode}")
    print(f"  Colormap: {args.cmap_style} (paper)")
    print(f"  Attention clipping: [{args.low_clip_percentile}, {args.clip_percentile}]%")
    print(f"  Gamma: {args.gamma}")
    print(f"  Canvas scale: {args.canvas_scale}, smooth: {args.smooth_sigma}")
    print(f"  Tissue outline: {args.show_tissue_outline}")
    print(f"  Top-k supernodes: {args.top_k_supernodes if args.show_topk_supernodes else 'off'}")
    if background_image_path:
        print(f"  Background image: {background_image_path}")
    
    # Process
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
        background_image_path=background_image_path,
        cmap_style=args.cmap_style,
        clip_percentile=args.clip_percentile,
        low_clip_percentile=args.low_clip_percentile,
        gamma=args.gamma,
        smooth_sigma=args.smooth_sigma,
        canvas_scale=args.canvas_scale,
        alpha=args.alpha,
        point_size=args.point_size,
        dpi=args.dpi,
        hide_axes=args.hide_axes,
        vis_mode=args.vis_mode,
        show_tissue_outline=args.show_tissue_outline,
        show_topk_supernodes=args.show_topk_supernodes,
        top_k_supernodes=args.top_k_supernodes,
        outline_color=args.outline_color,
        topk_outline_color=args.topk_outline_color,
        force_paper_heatmap=args.force_paper_heatmap,
        save_pdf=args.save_pdf,
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
        for t in alignment['sampled_tokens'][:5]:
            print(f"    idx={t['token_idx']}, raw={t['raw_idx']}, "
                  f"pos=({t['x']:.0f},{t['y']:.0f}), match={t['feature_match']}")
    
    print(f"\n  Outputs:")
    print(f"    Alignment report:  {result['alignment_report']}")
    print(f"    Debug heatmap:     {result['debug_heatmap_path']}")
    print(f"    Panel figure:      {result.get('panel_figure_path', 'N/A')}")
    print(f"    Paper heatmap:     {result.get('paper_heatmap_path', 'N/A')}")
    print(f"    Patch attention:   {result['patch_csv_path']}")
    print(f"    Supernode attn:    {result['supernode_csv_path']}")
    
    tissue = result['tissue_report']
    if tissue.get('warning'):
        print(f"\n  Tissue check: {tissue['warning']}")
    elif tissue.get('high_attn_tissue_ratio') is not None:
        print(f"\n  Tissue check: {tissue['high_attn_tissue_ratio']:.1%} high-attn on tissue")
    
    print(f"\n  Attention range: [{result['patch_attention'].min():.6f}, "
          f"{result['patch_attention'].max():.6f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate publication-quality topology attention heatmap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument('--slide_id', type=str, required=True)
    parser.add_argument('--coords_h5', type=str, required=True)
    parser.add_argument('--hilbert_pt', type=str, required=True)
    parser.add_argument('--feature_pt', type=str, required=True)
    
    # Alignment verification
    parser.add_argument('--raw_feature_pt', type=str, default=None)
    
    # Tissue mask
    parser.add_argument('--tissue_mask', type=str, default=None)
    
    # Background image
    parser.add_argument('--stitch_dir', type=str, default=None,
                        help='Directory with stitch images for background overlay')
    
    # Model architecture
    parser.add_argument('--checkpoint', type=str, default=None)
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
    parser.add_argument('--pool_mode', type=str, default='avg', choices=['avg', 'diffusion', 'residual', 'bp'])
    parser.add_argument('--diffusion_steps', type=int, default=0)
    parser.add_argument('--K_init', type=float, default=2.5)
    parser.add_argument('--atp_dt', type=float, default=0.1)
    parser.add_argument('--norm_type', type=str, default='mean', choices=['mean', 'sum'])
    parser.add_argument('--tau_init', type=float, default=2.0)
    parser.add_argument('--gamma_init', type=float, default=0.0)
    parser.add_argument('--bp_alpha_init', type=float, default=1.0)
    parser.add_argument('--bp_beta_init', type=float, default=1.0)
    parser.add_argument('--bp_lambda_init', type=float, default=1.0)
    
    # Segment config
    parser.add_argument('--local_segment_mode', type=str, default='none', choices=['none', 'chunk'])
    parser.add_argument('--local_segment_size', type=int, default=50)
    
    # Attention readout
    parser.add_argument('--attn_type', type=str, default='simple', choices=['simple', 'gated'])
    parser.add_argument('--attn_dim', type=int, default=128)
    
    # Attention mapping
    parser.add_argument('--attention_mapping', type=str, default='assign', choices=['assign', 'distribute'])
    
    # ── Visualization mode ──
    parser.add_argument('--vis_mode', type=str, default='panel',
                        choices=['scatter', 'raster', 'panel'],
                        help='Visualization mode: scatter (debug), raster (2D canvas), panel (3-panel figure)')
    
    # ── Colormap and normalization ──
    parser.add_argument('--cmap_style', type=str, default='soft_pathology',
                        choices=['soft_pathology', 'magma', 'YlOrRd', 'viridis', 'jet_debug', 'jet'],
                        help='Colormap style (jet not recommended for paper)')
    parser.add_argument('--low_clip_percentile', type=float, default=1,
                        help='Lower percentile for attention clipping (0-100)')
    parser.add_argument('--clip_percentile', type=float, default=99,
                        help='Upper percentile for attention clipping (0-100)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Gamma correction exponent (0-1, lower=more contrast)')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='Gaussian smoothing sigma for raster canvas (0=disabled)')
    parser.add_argument('--canvas_scale', type=int, default=32,
                        help='Canvas downscale factor for raster mode')
    
    # ── Point style ──
    parser.add_argument('--alpha', type=float, default=0.65, help='Point transparency')
    parser.add_argument('--point_size', type=float, default=1.5, help='Scatter point size')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--hide_axes', action='store_true', default=True, help='Hide axis labels')
    
    # ── Tissue outline ──
    parser.add_argument('--show_tissue_outline', action='store_true', default=True)
    parser.add_argument('--no_tissue_outline', dest='show_tissue_outline', action='store_false')
    parser.add_argument('--outline_color', type=str, default='#31a354')
    parser.add_argument('--outline_width', type=float, default=1.0)
    
    # ── Top supernodes ──
    parser.add_argument('--show_topk_supernodes', action='store_true', default=True)
    parser.add_argument('--no_topk_supernodes', dest='show_topk_supernodes', action='store_false')
    parser.add_argument('--top_k_supernodes', type=int, default=10)
    parser.add_argument('--topk_outline_color', type=str, default='#238b45')
    
    # ── Output ──
    parser.add_argument('--save_pdf', action='store_true', default=True)
    parser.add_argument('--no_save_pdf', dest='save_pdf', action='store_false')
    parser.add_argument('--force_paper_heatmap', action='store_true', default=False,
                        help='Generate paper heatmap even if alignment fails')
    
    # Mode
    parser.add_argument('--survival', action='store_true')
    parser.add_argument('--allow_partial_load', action='store_true')
    
    # Output dir
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join('heatmap_output', args.slide_id)
    
    main(args)
