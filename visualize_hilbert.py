import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.gridspec import GridSpec
import os


# ==========================================
# ==========================================
class PaperPalette:
    BG_COLOR = '#FFFFFF'
    PATCH_BASE = '#E0E0E0'
    RASTER_LINE = '#E65100'
    HILBERT_LINE = '#00897B'
    TEXT_COLOR = '#212121'


# ==========================================
# ==========================================
def print_quantitative_metrics(dist_default, dist_hilbert, tear_threshold):
    """
    计算并打印空间局部性的量化指标。
    返回 metrics 字典，供后续绘图调用。
    """
    metrics = {
        "Mean Jump Distance": [np.mean(dist_default), np.mean(dist_hilbert)],
        "Max Jump Distance (Worst Tear)": [np.max(dist_default), np.max(dist_hilbert)],
        f"Spatial Tears (Jumps > {int(tear_threshold)} px)": [
            np.sum(dist_default > tear_threshold),
            np.sum(dist_hilbert > tear_threshold)
        ],
        "Tear Rate (%)": [
            np.sum(dist_default > tear_threshold) / len(dist_default) * 100,
            np.sum(dist_hilbert > tear_threshold) / len(dist_hilbert) * 100
        ]
    }

    # 打印极其专业的控制台表格
    print("=" * 65)
    print(f"{'Quantitative Spatial Locality Metrics':^65}")
    print("=" * 65)
    print(f"{'Metric':<40} | {'Raster (Old)':<10} | {'Hilbert (Ours)':<10}")
    print("-" * 65)
    for k, v in metrics.items():
        if "Rate" in k:
            print(f"{k:<40} | {v[0]:>9.2f}% | {v[1]:>9.2f}%")
        else:
            print(f"{k:<40} | {int(v[0]):>10} | {int(v[1]):>10}")
    print("=" * 65)
    print(
        f"IHG-Mamba 将空间撕裂率从 {metrics['Tear Rate (%)'][0]:.2f}% 降至 {metrics['Tear Rate (%)'][1]:.2f}%！\n")

    return metrics


# ==========================================
# ==========================================
def aesthetic_quantify_visualize_v3(h5_path, hilbert_pt_path, save_traj="aesthetic_trajectory.png",
                                    save_hist="aesthetic_distance_hist.png"):
    print(f"正在读取坐标文件: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]

    num_patches = len(coords)
    print(f"共找到 {num_patches} 个 Patch 图块。\n")

    hilbert_idx = torch.load(hilbert_pt_path).numpy()

    # 1. 提取坐标顺序
    coords_default = coords
    coords_hilbert = coords[hilbert_idx]

    # 2. 计算物理跳跃距离
    dist_default = np.linalg.norm(coords_default[1:] - coords_default[:-1], axis=1)
    dist_hilbert = np.linalg.norm(coords_hilbert[1:] - coords_hilbert[:-1], axis=1)

    base_patch_size = np.min(dist_default[dist_default > 0])
    tear_threshold = base_patch_size * 3

    # 3. 调用封装好的方法打印并获取量化指标
    metrics = print_quantitative_metrics(dist_default, dist_hilbert, tear_threshold)

    # ==========================================
    # 🌟 绘图 1: 轨迹连线图
    # ==========================================
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), facecolor=PaperPalette.BG_COLOR)

    x_coords, y_coords = coords[:, 0], -coords[:, 1]

    def plot_beautiful_indices(ax, indices, line_color, title):
        ax.scatter(x_coords, y_coords, color=PaperPalette.PATCH_BASE, s=2, alpha=0.5, edgecolors='none', zorder=1)
        x_sorted, y_sorted = coords[indices, 0], -coords[indices, 1]

        pe = [PathEffects.withStroke(linewidth=2, foreground=line_color, alpha=0.2), PathEffects.Normal()]
        ax.plot(x_sorted, y_sorted, color=line_color, linewidth=0.6, alpha=0.8, path_effects=pe, zorder=2)

        ax.scatter(x_sorted[0], y_sorted[0], color='#4CAF50', s=60, marker='o', edgecolors='white', linewidth=1.5,
                   label='START', zorder=3)
        ax.scatter(x_sorted[-1], y_sorted[-1], color='#F44336', s=60, marker='X', edgecolors='white', linewidth=1.5,
                   label='END', zorder=3)

        ax.set_title(title, fontsize=16, fontweight='bold', color=PaperPalette.TEXT_COLOR, pad=15)
        ax.axis('equal');
        ax.axis('off')

        # 动态计算子图上的标注框数据
        max_jump = np.max(np.linalg.norm(coords[indices][1:] - coords[indices][:-1], axis=1))
        tears = np.sum(np.linalg.norm(coords[indices][1:] - coords[indices][:-1], axis=1) > tear_threshold)
        info_text = f"Max Jump: {int(max_jump)} px\nTears (>3x PS): {tears}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # 画左图 (Raster)
    plot_beautiful_indices(axes[0], np.arange(num_patches), PaperPalette.RASTER_LINE,
                           "Raster Scan Sequence (Baseline)\nSpatial locality is frequently disrupted")
    axes[0].legend(loc='upper right', fontsize=11, framealpha=0.8, edgecolor='none')

    # 画右图 (Hilbert)
    plot_beautiful_indices(axes[1], hilbert_idx, PaperPalette.HILBERT_LINE,
                           "Hilbert Curve Sequence (Ours)\nStrict topological boundary preservation")

    plt.suptitle("WSI Sequence Trajectory & Spatial Topology Comparison",
                 fontsize=22, fontweight='bold', color=PaperPalette.TEXT_COLOR, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig(save_traj, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"轨迹图已保存: {save_traj}")

    # ==========================================
    # 🌟 绘图 2: 直方图
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PaperPalette.BG_COLOR)
    bins = np.logspace(np.log10(base_patch_size * 0.9), np.log10(np.max(dist_default)), 50)

    ax.hist(dist_default, bins=bins, alpha=0.5, label='Raster Scan (Original)', color=PaperPalette.RASTER_LINE)
    ax.hist(dist_hilbert, bins=bins, alpha=0.7, label='Hilbert Curve (Ours)', color=PaperPalette.HILBERT_LINE)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical Distance between Adjacent Tokens (Pixels, Log Scale)', fontsize=14,
                  color=PaperPalette.TEXT_COLOR)
    ax.set_ylabel('Frequency (Log Scale)', fontsize=14, color=PaperPalette.TEXT_COLOR)
    ax.set_title('Distribution of Token-to-Token Spatial Jumps', fontsize=16, fontweight='bold',
                 color=PaperPalette.TEXT_COLOR)
    ax.axvline(tear_threshold, color='#212121', linestyle='dashed', linewidth=1.5, label='Spatial Tear Threshold')
    ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='#E0E0E0')
    ax.grid(True, which="both", ls="-", color='#EEEEEE', alpha=0.5)

    plt.savefig(save_hist, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"直方图已保存: {save_hist}")
    plt.show()


if __name__ == "__main__":

    H5_FILE = r"J:\Work\CLAM-master\toy_test\patches\macenko_demo_1.h5"
    HILBERT_PT = r"J:\Work\CLAM-master\toy_test\hilbert\macenko_demo_1_hilbert.pt"

    aesthetic_quantify_visualize_v3(H5_FILE, HILBERT_PT)