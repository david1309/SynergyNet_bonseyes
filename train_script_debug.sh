#!/usr/bin/env bash
#"LFPW,LFPW_Flip,HELEN,HELEN_Flip"
# AFW,AFW_Flip
#"IBUG,IBUG_Flip"

python3 main_train.py \
    --datatool-root-dir="/root/output_debug_all_wv" \
    --train-tags="IBUG" \
    --val-tags="IBUG_Flip" \
    \
    --debug=True \
    --exp-name="debug" \
    --use-cuda=True \
    --crop-images=False \
    \
    --epochs=10 \
    --batch-size=32 \
    --base-lr=0.0001\
    --milestones=5,8,9 \
    --save-val-freq=2 \
    --num-lms=77 \
    \
    --arch="mobilenet_v2" \
    --use-rot-inv=True \
    \
    --start-epoch=1 \
    --warmup=-1 \
    --print-freq=10 \
    --workers=4 \
    --resume="" \