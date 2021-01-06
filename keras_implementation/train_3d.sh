#!/usr/bin/env bash
# Used to train myDenoiser

### SUBJ1 ###
# Train AllButsubj1
printf "Training AllButsubj1Trained model...\n"
python scripts/train.py \
    --train_data="data/subj2/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
    --train_data="data/subj5/train" --train_data="data/subj6/train" \
    --val_data="data/subj2/train" --result_dir="3d_models/AllButsubj1Trained" --is_3d=True
#############

#### TODO: DELETE THIS ###
#python scripts/train.py --noise_level="all" \
#    --train_data="data/subj2/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj2/train" --result_dir="3d_models/AllButsubj1Trained"
##########################

### SUBJ2 ###
#printf "Training AllButsubj2Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj3/train" --result_dir="psnr_noise_estimated_models/AllButsubj2Trained" --is_3d=True
#############

### SUBJ3 ###
#printf "Training AllButsubj3Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj3Trained" --is_3d=True
#############

### SUBJ4 ###
#printf "Training AllButsubj4Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj5/train" --result_dir="psnr_noise_estimated_models/AllButsubj4Trained" --is_3d=True
#############

### SUBJ5 ###
#printf "Training AllButsubj5Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj5Trained" --is_3d=True
#############


### SUBJ6 ###
#printf "Training AllButsubj6Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj5/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj6Trained" --is_3d=True
#############
