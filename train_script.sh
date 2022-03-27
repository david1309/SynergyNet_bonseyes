#!/usr/bin/env bash

python3 main_train.py \
    --datatool-root-dir="/hdd1/datasets/300W_LP/output_debug_all/" \
    --train-tags="LFPW,LFPW_Flip,HELEN,HELEN_Flip" \
    --val-tags="IBUG,IBUG_Flip,AFW,AFW_Flip" \
    --debug=False \
    --exp-name="lm_77_all_data" \
    --use-cuda=True \
    \
    --epochs=20 \
    --batch-size=16 \
    --base-lr=0.00001\
    --milestones=10,15 \
    --save_val_freq=5 \
    --num-lms=77
    \
    --arch="mobilenet_v2" \
    \
    --start-epoch=1 \
    --warmup=5 \
    --print-freq=20 \
    --workers=4 \
    --test_initial=False \
    --resume="" \