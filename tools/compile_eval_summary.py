#!/usr/bin/env python3
"""Generate eval artifacts for all experiments and compile summary."""
import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS = {
    'hilbert_avg': {
        'results_dir': 'experiments/cls/LUAD_LUSC_hilbert_avg',
        'extra_args': ['--features_already_hilbert', '--use_hilbert_index', '--pool_size', '50', '--pool_mode', 'avg', '--K_init', '2.5', '--norm_type', 'mean', '--sampling_mode', 'random_points', '--order_mode', 'keep'],
    },
    'same_repo_baseline': {
        'results_dir': 'experiments/cls/LUAD_LUSC_same_repo_baseline',
        'extra_args': ['--features_not_hilbert', '--disable_atp_pool', '--local_layers', '1', '--global_layers', '0'],
    },
    'randomperm_avg': {
        'results_dir': 'experiments/cls/LUAD_LUSC_randomperm_avg',
        'extra_args': ['--features_already_hilbert', '--use_hilbert_index', '--pool_size', '50', '--pool_mode', 'avg', '--K_init', '2.5', '--norm_type', 'mean', '--sampling_mode', 'random_points', '--order_mode', 'random_perm', '--order_seed', '1'],
    },
    'raw_avg': {
        'results_dir': 'experiments/cls/LUAD_LUSC_raw_avg',
        'extra_args': ['--features_not_hilbert', '--pool_size', '50', '--pool_mode', 'avg', '--K_init', '2.5', '--norm_type', 'mean', '--sampling_mode', 'random_points', '--order_mode', 'keep'],
    },
    'hilbert_brpool': {
        'results_dir': 'experiments/cls/LUAD_LUSC_hilbert_brpool',
        'extra_args': ['--features_already_hilbert', '--use_hilbert_index', '--pool_size', '50', '--pool_mode', 'residual', '--gamma_init', '0.05', '--tau_init', '2.0', '--K_init', '2.5', '--norm_type', 'mean', '--sampling_mode', 'random_points', '--order_mode', 'keep'],
    },
}

DATA_ROOT = '/home/a255372639/LUSC_LUAD/ihg_data'
CSV_PATH = 'dataset_csv/LUAD_LUSC.csv'
SPLIT_DIR = 'splits/LUAD_LUSC_100'
FOLD = 0  # Use fold 0 for quick eval

def find_checkpoint(results_dir, fold=0):
    import glob
    pattern = os.path.join(results_dir, 'mamba_mil', '*', f's_{fold}_checkpoint.pt')
    files = glob.glob(pattern)
    return files[0] if files else None

def main():
    all_metrics = []
    
    for exp_name, exp_cfg in EXPERIMENTS.items():
        ckpt = find_checkpoint(exp_cfg['results_dir'], FOLD)
        if not ckpt:
            print(f"[SKIP] {exp_name}: no checkpoint found")
            continue
        
        output_dir = os.path.join(exp_cfg['results_dir'], 'eval_artifacts')
        if os.path.exists(os.path.join(output_dir, f'fold_{FOLD}', 'test_metrics.json')):
            print(f"[SKIP] {exp_name}: eval artifacts already exist")
            with open(os.path.join(output_dir, f'fold_{FOLD}', 'test_metrics.json')) as f:
                metrics = json.load(f)
            metrics['experiment'] = exp_name
            all_metrics.append(metrics)
            continue
        
        print(f"\n{'='*60}")
        print(f"  {exp_name}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, 'tools/generate_eval_from_checkpoint.py',
            '--checkpoint', ckpt,
            '--csv_path', CSV_PATH,
            '--split_dir', SPLIT_DIR,
            '--data_root_dir', DATA_ROOT,
            '--fold', str(FOLD),
            '--output_dir', output_dir,
            '--hidden_dim', '256',
            '--local_layers', '1',
            '--global_layers', '1',
            '--attn_type', 'simple',
            '--attn_dim', '128',
        ] + exp_cfg['extra_args']
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] {exp_name}: {result.stderr[-300:]}")
            continue
        
        metrics_path = os.path.join(output_dir, f'fold_{FOLD}', 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics['experiment'] = exp_name
            all_metrics.append(metrics)
    
    # Save summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        cols = ['experiment', 'auc', 'accuracy', 'balanced_accuracy', 
                'sensitivity', 'specificity', 'f1', 'precision', 'recall']
        cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df[cols]
        
        os.makedirs('experiments/eval_summary', exist_ok=True)
        summary_df.to_csv('experiments/eval_summary/all_experiments_metrics.csv', index=False)
        print(f"\n{'='*60}")
        print("Summary saved to: experiments/eval_summary/all_experiments_metrics.csv")
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
