#!/usr/bin/env bash
source /home/a255372639/miniconda3/etc/profile.d/conda.sh
conda activate ihg-mamba

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
cd /home/a255372639/projects/MambaMIL-main

DATA_ROOT="/home/a255372639/TCGA-LUAD/LUAD_example"
RESULTS_DIR="./experiments/train/ihg_ps50_K2.5_esfix"
LOG_DIR="./logs"
mkdir -p $RESULTS_DIR $LOG_DIR

echo "=========================================="
echo "  IHG-Mamba ps50_K2.5 esfix - START"
echo "  $(date)"
echo "=========================================="

python main_survival.py \
    --drop_out 0.5 \
    --early_stopping \
    --lr 1e-4 \
    --reg 1e-4 \
    --k 5 \
    --k_start -1 \
    --k_end -1 \
    --max_epochs 50 \
    --label_frac 1.0 \
    --exp_code "mamba_mil/uni" \
    --patch_size 512 \
    --batch_size 1 \
    --weighted_sample \
    --bag_loss nll_surv \
    --task "TCGA_LUAD_survival" \
    --backbone uni \
    --results_dir $RESULTS_DIR \
    --model_type mamba_mil \
    --split_dir "./splits/TCGA_LUAD_survival_kfold" \
    --data_root_dir $DATA_ROOT \
    --preloading no \
    --in_dim 1024 \
    --k_fold \
    --mambamil_rate 5 \
    --mambamil_layer 1 \
    --mambamil_type SRMamba \
    --seed 1 \
    --csv_path "dataset_csv/LUAD_processed.csv" \
    --hidden_dim 256 \
    --max_seq_len 2500 \
    --pool_size 50 \
    --local_layers 1 \
    --global_layers 1 \
    --diffusion_steps 2 \
    --K_init 2.5 \
    --atp_dt 0.1 \
    --norm_type mean \
    --features_already_hilbert \
    --es_warmup 0 \
    --es_patience 10 \
    --es_stop_epoch 10

echo "=========================================="
echo "  Training Complete! $(date)"
echo "=========================================="
