#!/usr/bin/env bash
# Used simply to test patch mixing (to reduce checkerboard effects)

# subj1
python scripts/inference.py \
    --set_dir="data/subj3" --model_dir_all_noise="models/AllButsubj3Trained/MyDnCNN_all_noise" --save_result=0
