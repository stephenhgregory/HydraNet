#!/usr/bin/env bash
# Used to run DnCNN denoising (inference) on all subj1-subj6

### SUBJ1 ###
# Run inference for subj1
printf "Calling test.py for model trained on all but subj1...\n"
python scripts/inference.py \
    --set_dir="dncnn_data/subj1" \
    --result_dir="dncnn_data/subj1_results" \
    --model_dir_dncnn="dncnn_models/AllButsubj1Trained/MyDnCNN_cleanup" \
    --skip_patch_denoise=1 --dncnn_denoise=1
#############

#### SUBJ2 ###
## Run inference for subj2
#printf "Calling test.py for model trained on all but subj2...\n"
#python scripts/inference.py \
#    --set_dir="dncnn_data/subj2" \
#    --result_dir="dncnn_data/subj2_results" \
#    --model_dir_dncnn="dncnn_models/AllButsubj2Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --dncnn_denoise=1
##############
#
#### SUBJ3 ###
## Run inference for subj3
#printf "Calling test.py for model trained on all but subj3...\n"
#python scripts/inference.py \
#    --set_dir="dncnn_data/subj3" \
#    --result_dir="dncnn_data/subj3_results" \
#    --model_dir_dncnn="dncnn_models/AllButsubj3Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --dncnn_denoise=1
##############
#
#### SUBJ4 ###
## Run inference for subj4
#printf "Calling test.py for model trained on all but subj4...\n"
#python scripts/inference.py \
#    --set_dir="dncnn_data/subj4" \
#    --result_dir="dncnn_data/subj4_results" \
#    --model_dir_dncnn="dncnn_models/AllButsubj4Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --dncnn_denoise=1
##############
#
#### SUBJ5 ###
## Run inference for subj5
#printf "Calling test.py for model trained on all but subj5...\n"
#python scripts/inference.py \
#    --set_dir="dncnn_data/subj5" \
#    --result_dir="dncnn_data/subj5_results" \
#    --model_dir_dncnn="dncnn_models/AllButsubj5Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --dncnn_denoise=1
##############
#
#### SUBJ6 ###
## Run inference for subj6
#printf "Calling test.py for model trained on all but subj6...\n"
#python scripts/inference.py \
#    --set_dir="dncnn_data/subj6" \
#    --result_dir="dncnn_data/subj6_results" \
#    --model_dir_dncnn="dncnn_models/AllButsubj6Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --dncnn_denoise=1
##############
#
#
