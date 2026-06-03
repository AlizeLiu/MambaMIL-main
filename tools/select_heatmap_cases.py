#!/usr/bin/env python3
"""Select representative cases for heatmaps from eval predictions."""
import os, sys, json, subprocess
import pandas as pd
import numpy as np

BASE = '/home/a255372639/projects/MambaMIL-main'
DATA_ROOT = '/home/a255372639/LUSC_LUAD/ihg_data'
PATCHES = '/home/a255372639/LUSC_LUAD/SCAD_example/patches'
HILBERT_IDX = '/home/a255372639/LUSC_LUAD/SCAD_example/hilbert'
HEATMAP_SCRIPT = os.path.join(BASE, 'tools/generate_topology_heatmap.py')
OUTPUT_BASE = os.path.join(BASE, 'experiments/heatmaps/selected_cases')
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Load predictions
h_pred = pd.read_csv(os.path.join(BASE, 'experiments/cls/LUAD_LUSC_hilbert_avg/eval_artifacts/fold_0/test_predictions.csv'))
r_pred_path = os.path.join(BASE, 'experiments/cls/LUAD_LUSC_randomperm_avg/eval_artifacts/fold_0/test_predictions.csv')
b_pred_path = os.path.join(BASE, 'experiments/cls/LUAD_LUSC_same_repo_baseline/eval_artifacts/fold_0/test_predictions.csv')

merged = h_pred.copy()
merged = merged.rename(columns={'pred': 'pred_hilbert', 'prob_1': 'prob_hilbert'})

if os.path.exists(r_pred_path):
    rp = pd.read_csv(r_pred_path)
    rp = rp.rename(columns={'pred': 'pred_random', 'prob_1': 'prob_random'})
    merged = merged.merge(rp[['slide_id', 'pred_random', 'prob_random']], on='slide_id', how='left')

if os.path.exists(b_pred_path):
    bp = pd.read_csv(b_pred_path)
    bp = bp.rename(columns={'pred': 'pred_baseline', 'prob_1': 'prob_baseline'})
    merged = merged.merge(bp[['slide_id', 'pred_baseline', 'prob_baseline']], on='slide_id', how='left')

print(f"Loaded {len(merged)} predictions with columns: {list(merged.columns)}")

# Select cases
selected = []

# 1. High confidence correct LUAD (prob_hilbert close to 0)
luad_correct = merged[(merged['label'] == 0) & (merged['pred_hilbert'] == 0)].sort_values('prob_hilbert')
for _, row in luad_correct.head(2).iterrows():
    selected.append({'slide_id': row['slide_id'], 'case_type': 'high_conf_LUAD', 'label': 0,
                     'pred_hilbert': row['pred_hilbert'], 'prob_hilbert': row['prob_hilbert']})

# 2. High confidence correct LUSC (prob_hilbert close to 1)
lusc_correct = merged[(merged['label'] == 1) & (merged['pred_hilbert'] == 1)].sort_values('prob_hilbert', ascending=False)
for _, row in lusc_correct.head(2).iterrows():
    selected.append({'slide_id': row['slide_id'], 'case_type': 'high_conf_LUSC', 'label': 1,
                     'pred_hilbert': row['pred_hilbert'], 'prob_hilbert': row['prob_hilbert']})

# 3. Hilbert correct / RandomPerm wrong
if 'pred_random' in merged.columns:
    h_right_r_wrong = merged[(merged['label'] == merged['pred_hilbert']) & (merged['label'] != merged['pred_random'])]
    for _, row in h_right_r_wrong.head(2).iterrows():
        selected.append({'slide_id': row['slide_id'], 'case_type': 'hilbert_right_random_wrong', 'label': row['label'],
                         'pred_hilbert': row['pred_hilbert'], 'prob_hilbert': row['prob_hilbert'],
                         'pred_random': row.get('pred_random'), 'prob_random': row.get('prob_random')})

# 4. Hilbert correct / Baseline wrong
if 'pred_baseline' in merged.columns:
    h_right_b_wrong = merged[(merged['label'] == merged['pred_hilbert']) & (merged['label'] != merged['pred_baseline'])]
    for _, row in h_right_b_wrong.head(2).iterrows():
        selected.append({'slide_id': row['slide_id'], 'case_type': 'hilbert_right_baseline_wrong', 'label': row['label'],
                         'pred_hilbert': row['pred_hilbert'], 'prob_hilbert': row['prob_hilbert'],
                         'pred_baseline': row.get('pred_baseline'), 'prob_baseline': row.get('prob_baseline')})

# 5. Hilbert wrong
h_wrong = merged[merged['label'] != merged['pred_hilbert']]
for _, row in h_wrong.head(2).iterrows():
    selected.append({'slide_id': row['slide_id'], 'case_type': 'hilbert_wrong', 'label': row['label'],
                     'pred_hilbert': row['pred_hilbert'], 'prob_hilbert': row['prob_hilbert']})

# Save case summary
case_df = pd.DataFrame(selected)
case_df.to_csv(os.path.join(OUTPUT_BASE, 'case_summary.csv'), index=False)
print(f"\nSelected {len(selected)} cases:")
for c in selected:
    print(f"  {c['case_type']}: {c['slide_id']} (label={c['label']}, prob={c['prob_hilbert']:.4f})")

# Generate heatmaps for each case
HEATMAP_CKPT = os.path.join(BASE, 'experiments/cls/LUAD_LUSC_hilbert_avg/mamba_mil/hilbert_avg_s1/s_0_checkpoint.pt')

for case in selected:
    sid = case['slide_id']
    case_type = case['case_type']
    out_dir = os.path.join(OUTPUT_BASE, case_type, sid)
    
    if os.path.exists(os.path.join(out_dir, 'topology_heatmap_scatter.png')):
        print(f"[SKIP] {case_type}/{sid}: heatmap already exists")
        continue
    
    print(f"\n[GENERATING] {case_type}/{sid}...")
    cmd = [
        'python', HEATMAP_SCRIPT,
        '--checkpoint', HEATMAP_CKPT,
        '--patch_dir', PATCHES,
        '--hilbert_idx_dir', HILBERT_IDX,
        '--feature_dir', os.path.join(DATA_ROOT, 'pt_files/uni'),
        '--slide_id', sid,
        '--output_dir', out_dir,
        '--hidden_dim', '256',
        '--pool_size', '50',
        '--pool_mode', 'avg',
        '--K_init', '2.5',
        '--norm_type', 'mean',
        '--attn_type', 'simple',
        '--attn_dim', '128',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    if result.returncode == 0:
        print(f"  OK: {out_dir}")
    else:
        print(f"  ERROR: {result.stderr[-200:]}")

print(f"\nDone! Output: {OUTPUT_BASE}/")
