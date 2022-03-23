#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="ckpts/logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/`date +'%Y-%m-%d_%H:%M.%S'`.log"

python3 main_train.py \
    --datatool_path="/hdd1/datasets/300W_LP/output_debug/IBUG" \
    --arch="mobilenet_v2" \
    --start-epoch=1 \
    --snapshot="ckpts/SynergyNet" \
    --warmup=5 \
    --batch-size=1024 \
    --base-lr=0.08 \
    --epochs=80 \
    --milestones=48,64 \
    --print-freq=50 \
    --devices-id=0 \
    --workers=4 \
    --log-file="${LOG_FILE}" \
    --test_initial=False \
    --save_val_freq=5 \
    --resume="" \
