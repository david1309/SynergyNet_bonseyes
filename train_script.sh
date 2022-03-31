#!/usr/bin/env bash
#"LFPW,LFPW_Flip,HELEN,HELEN_Flip"
# AFW,AFW_Flip
#"IBUG,IBUG_Flip"

python3 main_train.py \
    --datatool-root-dir="/root/300wlp/" \
    --train-tags="LFPW,LFPW_Flip,HELEN,HELEN_Flip" \
    --val-tags="IBUG,AFW" \
    \
    --debug=False \
    --exp-name="rot_inv_all_data" \
    --use-cuda=True \
    --crop-images=False \
    \
    --epochs=100 \
    --batch-size=128 \
    --base-lr=0.0001\
    --milestones=30,50,70,90 \
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