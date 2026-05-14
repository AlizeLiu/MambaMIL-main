#!/usr/bin/env bash
set -euo pipefail

# =========================
# MambaMIL LUAD Survival Analysis (IHG-Mamba)
# 5-Fold Cross Validation
# =========================

# 1. 激活 conda 环境
source /home/a255372639/miniconda3/etc/profile.d/conda.sh
conda activate ihg-mamba

# 2. 设置 GPU
export CUDA_VISIBLE_DEVICES=3

# 3. 设置工作目录
cd /home/a255372639/projects/MambaMIL-main

# 4. 设置路径参数（根据服务器实际路径修改）
# data_root_dir 下应有 pt_files/ 和 hilbert/ 两个子目录
DATA_ROOT="/home/a255372639/TCGA-LUAD/LUAD_example"
RESULTS_DIR="./experiments/train/TCGA_LUAD_survival"

# 5. 创建结果目录
mkdir -p $RESULTS_DIR

# 6. 超参数
LR="2e-4"
MAMBAMIL_RATE=5
MAMBAMIL_LAYER=2
MAMBAMIL_TYPE="SRMamba"
MODEL="mamba_mil"
BACKBONE="resnet50"
IN_DIM=1024
PATCH_SIZE="512"
K=5
SEED=1

# 7. 运行 5-fold 生存分析
echo "=========================================="
echo "  IHG-Mamba LUAD Survival Analysis"
echo "  Model: $MODEL ($MAMBAMIL_TYPE)"
echo "  Backbone: $BACKBONE"
echo "  Data: $DATA_ROOT"
echo "  Folds: $K"
echo "=========================================="

python main_survival.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr $LR \
    --k $K \
    --k_start -1 \
    --k_end -1 \
    --label_frac 1.0 \
    --exp_code "${MODEL}/${BACKBONE}" \
    --patch_size $PATCH_SIZE \
    --batch_size 1 \
    --weighted_sample \
    --bag_loss nll_surv \
    --task "TCGA_LUAD_survival" \
    --backbone $BACKBONE \
    --results_dir $RESULTS_DIR \
    --model_type $MODEL \
    --split_dir "./splits/TCGA_LUAD_survival_kfold" \
    --data_root_dir $DATA_ROOT \
    --preloading no \
    --in_dim $IN_DIM \
    --k_fold True \
    --mambamil_rate $MAMBAMIL_RATE \
    --mambamil_layer $MAMBAMIL_LAYER \
    --mambamil_type $MAMBAMIL_TYPE \
    --seed $SEED \
    --csv_path "dataset_csv/LUAD_processed.csv"

echo ""
echo "=========================================="
echo "  Training Complete!"
echo "  Results saved to: $RESULTS_DIR/${MODEL}_${BACKBONE}_s${SEED}/"
echo "=========================================="
