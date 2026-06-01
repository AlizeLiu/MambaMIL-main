import numpy as np
from scipy import stats

# Fold-level Test AUC data
hilbert_avg = np.array([0.9833, 0.9713, 0.9956, 0.9822, 0.9717])
same_repo_baseline = np.array([0.9666, 0.9568, 0.9960, 0.9906, 0.9760])
raw_avg = np.array([0.9684, 0.9684, 0.9956, 0.9826, 0.9611])
randomperm = np.array([0.9586, 0.9601, 0.9771, 0.9757, 0.9325])
brpool = np.array([0.9673, 0.9575, 0.9938, 0.9735, 0.9728])

pairs = [
    ("Hilbert+AvgPool", "Same-repo Baseline", hilbert_avg, same_repo_baseline),
    ("Hilbert+AvgPool", "Raw+AvgPool", hilbert_avg, raw_avg),
    ("Hilbert+AvgPool", "RandomPerm", hilbert_avg, randomperm),
    ("Hilbert+AvgPool", "BRPool", hilbert_avg, brpool),
]

print("=" * 80)
print("实验9: Paired Fold-Level 统计分析 (Test AUC)")
print("=" * 80)

for name_a, name_b, a, b in pairs:
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    t_stat, t_pval = stats.ttest_rel(a, b)
    try:
        w_stat, w_pval = stats.wilcoxon(diff)
    except:
        w_stat, w_pval = float('nan'), float('nan')
    n = len(diff)
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else "ns"
    print(f"\n{name_a} vs {name_b}:")
    print(f"  Paired delta: {mean_diff:+.4f} ± {std_diff:.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_pval:.4f} {sig}")
    print(f"  Wilcoxon: W={w_stat:.1f}, p={w_pval:.4f}")
    print(f"  95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
