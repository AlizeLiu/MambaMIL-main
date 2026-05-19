#!/usr/bin/env bash
source /home/a255372639/miniconda3/etc/profile.d/conda.sh
conda activate ihg-mamba
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
cd /home/a255372639/projects/MambaMIL-main

RESULTS_DIR="./experiments/train/IHG_chunk_ps50_BRPool_gamma005_hilbert_esfix"
mkdir -p $RESULTS_DIR

echo "=========================================="
echo "  Chunk+BRPool_gamma005 - START"
echo "  sampling_mode=chunk, chunk_size=50"
echo "  $(date)"
echo "=========================================="

python main_survival.py \
    --drop_out 0.5 --early_stopping --lr 1e-4 --reg 1e-4 \
    --k 5 --k_start -1 --k_end -1 --max_epochs 50 \
    --label_frac 1.0 --exp_code "mamba_mil/uni" --patch_size 512 \
    --batch_size 1 --weighted_sample --bag_loss nll_surv \
    --task "TCGA_LUAD_survival" --backbone uni \
    --results_dir $RESULTS_DIR \
    --model_type mamba_mil --k_fold \
    --split_dir "./splits/TCGA_LUAD_survival_kfold" \
    --data_root_dir "/home/a255372639/TCGA-LUAD/LUAD_example" \
    --preloading no --in_dim 1024 \
    --mambamil_rate 5 --mambamil_layer 1 --mambamil_type SRMamba \
    --seed 1 --csv_path "dataset_csv/LUAD_processed.csv" \
    --hidden_dim 256 --max_seq_len 2500 --pool_size 50 \
    --local_layers 1 --global_layers 1 --diffusion_steps 0 \
    --K_init 2.5 --atp_dt 0.1 --norm_type mean \
    --feature_subdir hilbert_pt \
    --features_already_hilbert \
    --sampling_mode chunk \
    --chunk_size 50 \
    --eval_chunk_strategy center \
    --pool_mode residual --tau_init 2.0 --gamma_init 0.05 \
    --es_warmup 0 --es_patience 10 --es_stop_epoch 10

echo "=========================================="
echo "  Chunk+BRPool_gamma005 Complete! $(date)"
echo "=========================================="
