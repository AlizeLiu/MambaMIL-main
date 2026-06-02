#!/usr/bin/env bash
set -euo pipefail

# =============================================
# Topology Heatmap Generation Example
# =============================================
# Usage: bash IHGTrain/generate_heatmap_example.sh
#
# This script generates attention heatmap for a single slide.
# Modify SLIDE_ID and CHECKPOINT_PATH before running.

source /home/a255372639/miniconda3/etc/profile.d/conda.sh
conda activate ihg-mamba

PROJECT_DIR="/home/a255372639/projects/MambaMIL-main"
cd "$PROJECT_DIR"

# ===== Modify these paths =====
SLIDE_ID="TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01"
CHECKPOINT_PATH="experiments/cls/LUAD_LUSC_hilbert_avg/mamba_mil/hilbert_avg_s1/s_0_checkpoint.pt"
OUTPUT_DIR="experiments/heatmaps/demo"

DATA_ROOT="/home/a255372639/LUSC_LUAD/ihg_data"
H5_DIR="/home/a255372639/LUSC_LUAD/SCAD_example/patches"
HILBERT_DIR="/home/a255372639/LUSC_LUAD/SCAD_example/hilbert"
CSV_PATH="dataset_csv/LUAD_LUSC.csv"
# ==============================

mkdir -p "$OUTPUT_DIR"

python tools/generate_topology_heatmap.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --slide_id "$SLIDE_ID" \
    --data_root_dir "$DATA_ROOT" \
    --feature_subdir pt_files \
    --backbone uni \
    --patch_size 512 \
    --h5_dir "$H5_DIR" \
    --hilbert_index_dir "$HILBERT_DIR" \
    --csv_path "$CSV_PATH" \
    --model_type mamba_mil \
    --in_dim 1024 \
    --n_classes 2 \
    --hidden_dim 256 \
    --pool_size 50 \
    --pool_mode avg \
    --attn_type simple \
    --attn_dim 128 \
    --local_layers 1 \
    --global_layers 1 \
    --mambamil_type SRMamba \
    --mambamil_rate 5 \
    --max_seq_len 999999 \
    --sampling_mode random_points \
    --order_mode keep \
    --output_dir "$OUTPUT_DIR" \
    --top_k_supernodes 10

echo "=========================================="
echo "  Heatmap saved to: $OUTPUT_DIR/$SLIDE_ID/"
echo "=========================================="
