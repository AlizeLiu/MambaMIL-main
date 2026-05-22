#!/bin/bash
# ==============================================================================
# IHG-Mamba Classification: LUAD_LUSC Baseline (Light)
# - No Hilbert ordering, standard pt_files features
# - No ATP-Pool (Identity fallback)
# - random_points sampling, max_seq_len=2500
# ==============================================================================

set -e

# ---- Paths (edit these) ----
DATA_ROOT_DIR="/path/to/your/LUAD_LUSC/data"   # <-- EDIT THIS
CSV_PATH="dataset_csv/LUAD_LUSC.csv"
SPLIT_DIR="/home/a255372639/baseline/MambaMIL-main/splits/LUAD_LUSC_100"

# ---- Task ----
TASK="LUAD_LUSC"
MODEL_TYPE="mamba_mil"
N_CLASSES=2
IN_DIM=1024

# ---- Model architecture ----
HIDDEN_DIM=256
LOCAL_LAYERS=1
GLOBAL_LAYERS=1
MAMBAMIL_RATE=5
MAMBAMIL_LAYER=1
MAMBAMIL_TYPE="SRMamba"

# ---- ATP-Pool (disabled for baseline) ----
DISABLE_ATP_POOL="--disable_atp_pool"
POOL_SIZE=50
POOL_MODE="avg"
DIFFUSION_STEPS=0
K_INIT=2.5
ATP_DT=0.1
NORM_TYPE="mean"
TAU_INIT=2.0
GAMMA_INIT=0.0

# ---- IHG features ----
FEATURE_SUBDIR="pt_files"
FEATURES_HILBERT="--features_not_hilbert"

# ---- Sampling ----
SAMPLING_MODE="random_points"
MAX_SEQ_LEN=2500
CHUNK_SIZE=50
EVAL_CHUNK_STRATEGY="center"
ORDER_MODE="keep"
ORDER_SEED=1

# ---- Local segment ----
LOCAL_SEGMENT_MODE="none"
LOCAL_SEGMENT_SIZE=50

# ---- Training ----
BATCH_SIZE=1
DROP_OUT=0.5
LR=1e-4
REG=1e-4
MAX_EPOCHS=200
SEED=1
K=10

# ---- Early stopping ----
EARLY_STOPPING="--early_stopping"
ES_PATIENCE=20
ES_STOP_EPOCH=20

# ---- Results ----
EXP_CODE="cls_luad_lusc_baseline_light"
RESULTS_DIR="./experiments/classification/${TASK}"

echo "============================================================"
echo " IHG-Mamba Classification: LUAD_LUSC Baseline (Light)"
echo "============================================================"
echo " data_root_dir: ${DATA_ROOT_DIR}"
echo " feature_subdir: ${FEATURE_SUBDIR}"
echo " atp_pool: DISABLED"
echo "============================================================"

python main.py \
    --task ${TASK} \
    --model_type ${MODEL_TYPE} \
    --in_dim ${IN_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --local_layers ${LOCAL_LAYERS} \
    --global_layers ${GLOBAL_LAYERS} \
    --mambamil_rate ${MAMBAMIL_RATE} \
    --mambamil_layer ${MAMBAMIL_LAYER} \
    --mambamil_type ${MAMBAMIL_TYPE} \
    ${DISABLE_ATP_POOL} \
    --pool_size ${POOL_SIZE} \
    --pool_mode ${POOL_MODE} \
    --diffusion_steps ${DIFFUSION_STEPS} \
    --K_init ${K_INIT} \
    --atp_dt ${ATP_DT} \
    --norm_type ${NORM_TYPE} \
    --tau_init ${TAU_INIT} \
    --gamma_init ${GAMMA_INIT} \
    --feature_subdir ${FEATURE_SUBDIR} \
    ${FEATURES_HILBERT} \
    --sampling_mode ${SAMPLING_MODE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --chunk_size ${CHUNK_SIZE} \
    --eval_chunk_strategy ${EVAL_CHUNK_STRATEGY} \
    --order_mode ${ORDER_MODE} \
    --order_seed ${ORDER_SEED} \
    --local_segment_mode ${LOCAL_SEGMENT_MODE} \
    --local_segment_size ${LOCAL_SEGMENT_SIZE} \
    --data_root_dir ${DATA_ROOT_DIR} \
    --csv_path ${CSV_PATH} \
    --split_dir ${SPLIT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --drop_out ${DROP_OUT} \
    --lr ${LR} \
    --reg ${REG} \
    --max_epochs ${MAX_EPOCHS} \
    --seed ${SEED} \
    --k ${K} \
    --exp_code ${EXP_CODE} \
    --results_dir ${RESULTS_DIR} \
    ${EARLY_STOPPING} \
    --es_patience ${ES_PATIENCE} \
    --es_stop_epoch ${ES_STOP_EPOCH}
