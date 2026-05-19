     1|#!/usr/bin/env bash
     2|source /home/a255372639/miniconda3/etc/profile.d/conda.sh
     3|conda activate ihg-mamba
     4|
     5|export CUDA_VISIBLE_DEVICES=1
     6|export PYTHONUNBUFFERED=1
     7|export WANDB_MODE=offline
     8|cd /home/a255372639/projects/MambaMIL-main
     9|
    10|DATA_ROOT="/home/a255372639/TCGA-LUAD/LUAD_example"
    11|RESULTS_DIR="./experiments/train/TCGA_LUAD_survival"
    12|mkdir -p $RESULTS_DIR
    13|
    14|echo "=========================================="
    15|echo "  IHG-Mamba LUAD Survival Analysis (pool_size=50) - START"
    16|echo "  $(date)"
    17|echo "=========================================="
    18|
    19|python main_survival.py \
    20|    --drop_out 0.5 \
    21|    --early_stopping \
    22|    --lr 1e-4 \
    23|    --reg 1e-4 \
    24|    --k 5 \
    25|    --k_start -1 \
    26|    --k_end -1 \
    27|    --max_epochs 50 \
    28|    --label_frac 1.0 \
    29|    --exp_code "mamba_mil/uni" \
    30|    --patch_size 512 \
    31|    --batch_size 1 \
    32|    --weighted_sample \
    33|    --bag_loss nll_surv \
    34|    --task "TCGA_LUAD_survival" \
    35|    --backbone uni \
    36|    --results_dir $RESULTS_DIR \
    37|    --model_type mamba_mil \
    38|    --split_dir "./splits/TCGA_LUAD_survival_kfold" \
    39|    --data_root_dir $DATA_ROOT \
    40|    --preloading no \
    41|    --in_dim 1024 \
    42|    --k_fold \
    43|    --mambamil_rate 5 \
    44|    --mambamil_layer 1 \
    45|    --mambamil_type SRMamba \
    46|    --seed 1 \
    47|    --csv_path "dataset_csv/LUAD_processed.csv" \
    48|    --hidden_dim 256 \
    49|    --max_seq_len 2500 \
    50|    --pool_size 50 \
    51|    --local_layers 1 \
    52|    --global_layers 1 \
    53|    --diffusion_steps 2 \
    54|    --K_init 0.5 \
    55|    --atp_dt 0.1 \
    56|    --features_not_hilbert \
    --use_hilbert_index \
    57|    --es_warmup 5 \
    58|    --es_patience 5 \
    59|    --es_stop_epoch 5
    60|
    61|echo "=========================================="
    62|echo "  Training Complete! $(date)"
    63|echo "=========================================="
    64|