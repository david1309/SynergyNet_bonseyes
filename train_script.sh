#!/usr/bin/env bash
#"LFPW,LFPW_Flip,HELEN,HELEN_Flip"
#"IBUG,IBUG_Flip,AFW,AFW_Flip"

python3 main_train.py \
    --datatool-root-dir="/hdd1/datasets/300W_LP/output_debug_all/" \
    --train-tags="HELEN" \
    --val-tags="IBUG" \
    \
    --debug=False \
    --exp-name="loss_weights_100_HELEN_crop" \
    --use-cuda=True \
    --crop-images=False \
    \
    --epochs=10 \
    --batch-size=16 \
    --base-lr=0.00001\
    --milestones=6,8 \
    --save-val-freq=2 \
    --num-lms=77 \
    \
    --arch="mobilenet_v2" \
    \
    --start-epoch=1 \
    --warmup=-1 \
    --print-freq=10 \
    --workers=4 \
    --resume="" \