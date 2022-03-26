#!/usr/bin/env bash

python3 main_train.py \
    --datatool-root-dir="/hdd1/datasets/300W_LP/output_debug/" \
    --train-tags="AFW" \
    --val-tags="IBUG" \
    --debug=False \
    \
    --batch-size=16 \
    --base-lr=0.00001\
    --epochs=100 \
    --milestones=48,64 \
    --save_val_freq=5 \
    --num-lms=77
    \
    --arch="mobilenet_v2" \
    \
    --start-epoch=1 \
    --warmup=5 \
    --print-freq=20 \
    --devices-id=0 \
    --workers=4 \
    --test_initial=False \
    --resume="" \
