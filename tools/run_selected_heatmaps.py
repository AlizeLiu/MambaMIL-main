#!/usr/bin/env python3
"""Generate heatmaps for selected representative cases."""
import os, sys, subprocess, pandas as pd

BASE = '/home/a255372639/projects/MambaMIL-main'
PATCHES = '/home/a255372639/LUSC_LUAD/SCAD_example/patches'
HILBERT_IDX = '/home/a255372639/LUSC_LUAD/SCAD_example/hilbert'
FEATURE_DIR = '/home/a255372639/LUSC_LUAD/ihg_data/pt_files/uni'
HEATMAP_SCRIPT = os.path.join(BASE, 'tools/generate_topology_heatmap.py')
OUTPUT_BASE = os.path.join(BASE, 'experiments/heatmaps/selected_cases')
HEATMAP_CKPT = os.path.join(BASE, 'experiments/cls/LUAD_LUSC_hilbert_avg/mamba_mil/hilbert_avg_s1/s_0_checkpoint.pt')

case_df = pd.read_csv(os.path.join(OUTPUT_BASE, 'case_summary.csv'))
print(f'Processing {len(case_df)} cases...')

for _, case in case_df.iterrows():
    sid = case['slide_id']
    case_type = case['case_type']
    out_dir = os.path.join(OUTPUT_BASE, case_type, sid)
    
    if os.path.exists(os.path.join(out_dir, 'topology_heatmap_scatter.png')):
        print(f'[SKIP] {case_type}/{sid}')
        continue
    
    coords_h5 = os.path.join(PATCHES, f'{sid}.h5')
    hilbert_pt = os.path.join(HILBERT_IDX, f'{sid}_hilbert.pt')
    feature_pt = os.path.join(FEATURE_DIR, f'{sid}.pt')
    
    missing = [n for f, n in [(coords_h5,'coords'),(hilbert_pt,'hilbert'),(feature_pt,'feature')] if not os.path.exists(f)]
    if missing:
        print(f'[SKIP] {case_type}/{sid}: missing {missing}')
        continue
    
    print(f'[GENERATING] {case_type}/{sid}...')
    cmd = [
        'python', HEATMAP_SCRIPT,
        '--slide_id', sid,
        '--coords_h5', coords_h5,
        '--hilbert_pt', hilbert_pt,
        '--feature_pt', feature_pt,
        '--checkpoint', HEATMAP_CKPT,
        '--output_dir', out_dir,
        '--hidden_dim', '256', '--pool_size', '50', '--pool_mode', 'avg',
        '--K_init', '2.5', '--norm_type', 'mean',
        '--attn_type', 'simple', '--attn_dim', '128',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE, timeout=120)
    if result.returncode == 0:
        files = os.listdir(out_dir) if os.path.isdir(out_dir) else []
        print(f'  OK: {files}')
    else:
        print(f'  ERROR: {result.stderr[-300:]}')

print('\nDone!')
