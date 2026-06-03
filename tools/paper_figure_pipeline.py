#!/usr/bin/env python3
"""
Comprehensive paper-figure pipeline for IHG-Mamba LUAD/LUSC classification.
Generates: statistics, figures (bar, ROC, confusion, spatial), summary tables.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

BASE = '/home/a255372639/projects/MambaMIL-main'
OUT = os.path.join(BASE, 'experiments/paper_figures')
os.makedirs(OUT, exist_ok=True)


# ============================================================
# 1. Load all 5-fold summary.csv
# ============================================================
EXPERIMENTS = {
    'Hilbert+AvgPool':   'experiments/cls/LUAD_LUSC_hilbert_avg/mamba_mil/hilbert_avg_s1/summary.csv',
    'SameRepo-Baseline': 'experiments/cls/LUAD_LUSC_same_repo_baseline/mamba_mil/same_repo_baseline_s1/summary.csv',
    'RandomPerm':        'experiments/cls/LUAD_LUSC_randomperm_avg/mamba_mil/randomperm_avg_s1/summary.csv',
    'Raw+AvgPool':       'experiments/cls/LUAD_LUSC_raw_avg/mamba_mil/raw_avg_s1/summary.csv',
    'Hilbert+BRPool':    'experiments/cls/LUAD_LUSC_hilbert_brpool/mamba_mil/hilbert_brpool_s1/summary.csv',
    'Original+Hilbert':  'experiments/cls/LUAD_LUSC_original_hilbert/mamba_mil/original_hilbert_s1/summary.csv',
}

all_folds = {}
all_means = {}
for name, path in EXPERIMENTS.items():
    df = pd.read_csv(os.path.join(BASE, path))
    # Keep only fold rows (not mean/std)
    folds = df[df['folds'].apply(lambda x: str(x).isdigit())].copy()
    folds = folds.drop(columns=['folds'])
    folds = folds.apply(pd.to_numeric, errors='coerce')
    all_folds[name] = folds
    all_means[name] = {
        'test_auc_mean': folds['test_auc'].mean(),
        'test_auc_std': folds['test_auc'].std(),
        'test_acc_mean': folds['test_acc'].mean(),
        'test_acc_std': folds['test_acc'].std(),
        'val_auc_mean': folds['val_auc'].mean(),
        'val_auc_std': folds['val_auc'].std(),
    }

# ============================================================
# 2. Main results table
# ============================================================
print("=" * 80)
print("Table 1: Main Results (5-fold CV)")
print("=" * 80)
header = f"{'Experiment':<22} {'Test AUC':>18} {'Test Acc':>18} {'Val AUC':>18}"
print(header)
print("-" * len(header))
for name in EXPERIMENTS:
    m = all_means[name]
    print(f"{name:<22} {m['test_auc_mean']:.4f}±{m['test_auc_std']:.4f}  "
          f"{m['test_acc_mean']:.4f}±{m['test_acc_std']:.4f}  "
          f"{m['val_auc_mean']:.4f}±{m['val_auc_std']:.4f}")


# ============================================================
# 3. Paired statistics
# ============================================================
print("\n" + "=" * 80)
print("Table 2: Paired Statistics (vs Hilbert+AvgPool)")
print("=" * 80)
ref = all_folds['Hilbert+AvgPool']['test_auc'].values

stat_rows = []
for name in ['SameRepo-Baseline', 'Raw+AvgPool', 'RandomPerm', 'Hilbert+BRPool', 'Original+Hilbert']:
    comp = all_folds[name]['test_auc'].values
    delta = ref - comp
    t_stat, t_p = stats.ttest_rel(ref, comp)
    try:
        w_stat, w_p = stats.wilcoxon(ref, comp)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    ci95 = stats.t.interval(0.95, len(delta)-1, loc=delta.mean(), scale=stats.sem(delta))
    
    row = {
        'Comparison': f'Hilbert vs {name}',
        'Mean ΔAUC': f"{delta.mean():.4f}",
        '95% CI': f"[{ci95[0]:.4f}, {ci95[1]:.4f}]",
        'paired_t_p': f"{t_p:.4f}",
        'wilcoxon_p': f"{w_p:.4f}" if not np.isnan(w_p) else "N/A",
    }
    stat_rows.append(row)
    
    sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
    print(f"Hilbert vs {name:<20}: Δ={delta.mean():.4f}±{delta.std():.4f}, "
          f"t_p={t_p:.4f} {sig}, w_p={w_p:.4f}" if not np.isnan(w_p) else
          f"Hilbert vs {name:<20}: Δ={delta.mean():.4f}±{delta.std():.4f}, t_p={t_p:.4f} {sig}")

# Save statistics CSV
stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUT, 'paired_statistics.csv'), index=False)


# ============================================================
# 4. Figure: Performance bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

exp_names = list(EXPERIMENTS.keys())
auc_means = [all_means[n]['test_auc_mean'] for n in exp_names]
auc_stds = [all_means[n]['test_auc_std'] for n in exp_names]
acc_means = [all_means[n]['test_acc_mean'] for n in exp_names]
acc_stds = [all_means[n]['test_acc_std'] for n in exp_names]

colors = ['#2196F3', '#9E9E9E', '#F44336', '#FF9800', '#9C27B0', '#607D8B']
short_names = ['Hilbert\n+Avg', 'Baseline', 'Random\nPerm', 'Raw\n+Avg', 'Hilbert\n+BRPool', 'Original\n+Hilbert']

# AUC bar
ax = axes[0]
bars = ax.bar(short_names, auc_means, yerr=auc_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Test AUC', fontsize=12)
ax.set_title('Test AUC by Experiment', fontsize=13)
ax.set_ylim(0.94, 1.00)
for i, (m, s) in enumerate(zip(auc_means, auc_stds)):
    ax.text(i, m + s + 0.001, f'{m:.3f}', ha='center', fontsize=9)

# Accuracy bar
ax = axes[1]
bars = ax.bar(short_names, acc_means, yerr=acc_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Test Accuracy by Experiment', fontsize=13)
ax.set_ylim(0.88, 1.00)
for i, (m, s) in enumerate(zip(acc_means, acc_stds)):
    ax.text(i, m + s + 0.001, f'{m:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_performance_bars.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"\nFigure 2 saved: {OUT}/fig2_performance_bars.png")


# ============================================================
# 5. Figure: Spatial diagnostics
# ============================================================
# Load existing spatial diagnostics if available
diag_dir = os.path.join(BASE, 'experiments/diagnostics')
if os.path.isdir(diag_dir):
    diag_files = {f.replace('.csv', ''): os.path.join(diag_dir, f) 
                  for f in os.listdir(diag_dir) if f.endswith('.csv')}
    
    if diag_files:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        orders = []
        mean_jumps = []
        tear_rates = []
        
        for fname, fpath in sorted(diag_files.items()):
            df_diag = pd.read_csv(fpath)
            if 'mean_jump' in df_diag.columns:
                orders.append(fname.split('_')[-1].capitalize())
                mean_jumps.append(df_diag['mean_jump'].mean())
                if 'tear_rate' in df_diag.columns:
                    tear_rates.append(df_diag['tear_rate'].mean())
        
        if orders:
            x = range(len(orders))
            axes[0].bar(x, mean_jumps, color=['#4CAF50', '#2196F3', '#F44336'][:len(orders)])
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(orders)
            axes[0].set_ylabel('Mean Jump Distance')
            axes[0].set_title('Spatial Continuity (Jump Distance)')
            
            if tear_rates:
                axes[1].bar(x, tear_rates, color=['#4CAF50', '#2196F3', '#F44336'][:len(orders)])
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(orders)
                axes[1].set_ylabel('Tear Rate')
                axes[1].set_title('Spatial Discontinuity (Tear Rate)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'fig5_spatial_diagnostics.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Figure 5 saved: {OUT}/fig5_spatial_diagnostics.png")


# ============================================================
# 6. Figure: Fold-level ROC curves for key experiments
# ============================================================
# Load eval artifacts if available
eval_dir_template = os.path.join(BASE, 'experiments/cls/LUAD_LUSC_{exp}/eval_artifacts/fold_0')
key_exps = {
    'hilbert_avg': 'Hilbert+AvgPool',
    'same_repo_baseline': 'Baseline',
    'randomperm_avg': 'RandomPerm',
}

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
roc_colors = ['#2196F3', '#9E9E9E', '#F44336']

for (exp_key, label), color in zip(key_exps.items(), roc_colors):
    metrics_path = eval_dir_template.format(exp=exp_key)
    pred_file = os.path.join(metrics_path, 'test_predictions.csv')
    if os.path.exists(pred_file):
        pred_df = pd.read_csv(pred_file)
        if 'prob_1' in pred_df.columns and 'label' in pred_df.columns:
            fpr, tpr, _ = roc_curve(pred_df['label'], pred_df['prob_1'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves (Fold 0)', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_roc_curves.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Figure 3 saved: {OUT}/fig3_roc_curves.png")


# ============================================================
# 7. Summary CSV
# ============================================================
summary_rows = []
for name in EXPERIMENTS:
    m = all_means[name]
    summary_rows.append({
        'Experiment': name,
        'Test_AUC_mean': m['test_auc_mean'],
        'Test_AUC_std': m['test_auc_std'],
        'Test_Acc_mean': m['test_acc_mean'],
        'Test_Acc_std': m['test_acc_std'],
        'Val_AUC_mean': m['val_auc_mean'],
        'Val_AUC_std': m['val_auc_std'],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT, 'main_results_table.csv'), index=False)
print(f"\nMain results table saved: {OUT}/main_results_table.csv")


# ============================================================
# 8. Multi-seed comparison (if available)
# ============================================================
s2_path = os.path.join(BASE, 'experiments/cls/LUAD_LUSC_hilbert_avg_2/mamba_mil/hilbert_avg_2_s2/summary.csv')
if not os.path.exists(s2_path):
    s2_path = os.path.join(BASE, 'experiments/cls/LUAD_LUSC_hilbert_avg_s2/mamba_mil/hilbert_avg_s2_s2/summary.csv')

if os.path.exists(s2_path):
    df_s2 = pd.read_csv(s2_path)
    folds_s2 = df_s2[df_s2['folds'].apply(lambda x: str(x).isdigit())]
    folds_s2 = folds_s2.apply(pd.to_numeric, errors='coerce')
    s2_auc = folds_s2['test_auc'].mean()
    s2_std = folds_s2['test_auc'].std()
    print(f"\nMulti-seed: Hilbert+AvgPool seed=2: AUC={s2_auc:.4f}±{s2_std:.4f}")
    
    # Combined
    combined_auc = (all_folds['Hilbert+AvgPool']['test_auc'].mean() + s2_auc) / 2
    print(f"Combined multi-seed mean AUC: {combined_auc:.4f}")


# ============================================================
# 9. Eval artifacts summary (with sensitivity/specificity if available)
# ============================================================
eval_summary = []
for exp_key, label in key_exps.items():
    metrics_path = os.path.join(BASE, f'experiments/cls/LUAD_LUSC_{exp_key}/eval_artifacts/fold_0/test_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        eval_summary.append({
            'Experiment': label,
            'AUC': m.get('auc', 'N/A'),
            'Accuracy': m.get('accuracy', 'N/A'),
            'Sensitivity': m.get('sensitivity', 'N/A'),
            'Specificity': m.get('specificity', 'N/A'),
            'F1': m.get('f1', 'N/A'),
            'Balanced_Acc': m.get('balanced_accuracy', 'N/A'),
            'Precision': m.get('precision', 'N/A'),
        })

if eval_summary:
    eval_df = pd.DataFrame(eval_summary)
    eval_df.to_csv(os.path.join(OUT, 'eval_artifacts_summary.csv'), index=False)
    print(f"\nEval artifacts summary saved: {OUT}/eval_artifacts_summary.csv")
    print(eval_df.to_string(index=False))


print(f"\n{'='*80}")
print("Pipeline complete! Output directory:")
print(f"  {OUT}/")
print(f"  - main_results_table.csv")
print(f"  - paired_statistics.csv")
print(f"  - eval_artifacts_summary.csv")
print(f"  - fig2_performance_bars.png")
print(f"  - fig3_roc_curves.png")
print(f"  - fig5_spatial_diagnostics.png (if diag data exists)")
print(f"{'='*80}")
