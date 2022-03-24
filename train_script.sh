#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="ckpts/logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/`date +'%Y-%m-%d_%H:%M.%S'`.log"

python3 main_train.py \
    --datatool-root-dir="/hdd1/datasets/300W_LP/output_debug/" \
    --train-tags="AFW" \
    --val-tags="IBUG" \
    \
    --batch-size=16 \
    --base-lr=0.00001\
    --epochs=80 \
    --milestones=48,64 \
    --save_val_freq=5 \
    \
    --arch="mobilenet_v2" \
    --snapshot="ckpts/SynergyNet" \
    --log-file="${LOG_FILE}" \
    \
    --start-epoch=1 \
    --warmup=5 \
    --print-freq=20 \
    --devices-id=0 \
    --workers=4 \
    --test_initial=False \
    --resume="" \
