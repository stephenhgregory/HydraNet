#!/usr/bin/env bash
# Used to run inference on all subj1-subj6

### SUBJ1 ###
printf "Training AllButsubj1TrainedCleanup model...\n"
python scripts/train.py \
    --blurry_data="psnr_results/subj2_results/train" --blurry_data="psnr_results/subj3_results/train" \
    --blurry_data="psnr_results/subj4_results/train" --blurry_data="psnr_results/subj5_results/train" \
    --blurry_data="psnr_results/subj6_results/train" \
    --clear_data="data/subj2/train/ClearImages" --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj4/train/ClearImages" \
    --clear_data="data/subj5/train/ClearImages" --clear_data="data/subj6/train/ClearImages" \
    --val_data="psnr_results/subj1_results/train" --result_dir="psnr_noise_estimated_models/AllButsubj1Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj2/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj2/train" --result_dir="psnr_noise_estimated_models/AllButsubj1Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj2/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj2/train" --result_dir="psnr_noise_estimated_models/AllButsubj1Trained" --is_cleanup=True
### Run inference for subj1
##printf "Calling test.py for model trained on all but subj1...\n"
##python scripts/inference.py \
##    --single_denoiser=0 \
##    --set_dir="data/subj1" \
##    --train_data="data/subj2/train" \
##    --result_dir="psnr_results/subj1_results" \
##    --model_dir_all_noise="psnr_noise_estimated_models/AllButsubj1Trained/MyDnCNN_all_noise" \
##    --model_dir_low_noise="psnr_noise_estimated_models/AllButsubj1Trained/MyDnCNN_low_noise" \
##    --model_dir_medium_noise="psnr_noise_estimated_models/AllButsubj1Trained/MyDnCNN_medium_noise" \
##    --model_dir_high_noise="psnr_noise_estimated_models/AllButsubj1Trained/MyDnCNN_high_noise"
##############
#
#### SUBJ2 ###
#printf "Training AllButsubj2Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj3/train" --result_dir="psnr_noise_estimated_models/AllButsubj2Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj3/train" --result_dir="psnr_noise_estimated_models/AllButsubj2Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj3/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj3/train" --result_dir="psnr_noise_estimated_models/AllButsubj2Trained" --is_cleanup=True
### Run inference for subj2
##printf "Calling test.py for model trained on all but subj2...\n"
##python scripts/inference.py \
##    --single_denoiser=0 \
##    --set_dir="data/subj2" \
##    --train_data="data/subj3/train" \
##    --result_dir="psnr_results/subj2_results" \
##    --model_dir_all_noise="psnr_noise_estimated_models/AllButsubj2Trained/MyDnCNN_all_noise" \
##    --model_dir_low_noise="psnr_noise_estimated_models/AllButsubj2Trained/MyDnCNN_low_noise" \
##    --model_dir_medium_noise="psnr_noise_estimated_models/AllButsubj2Trained/MyDnCNN_medium_noise" \
##    --model_dir_high_noise="psnr_noise_estimated_models/AllButsubj2Trained/MyDnCNN_high_noise"
##############
#
#### SUBJ3 ###
#printf "Training AllButsubj3Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj3Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj3Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj4/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj3Trained" --is_cleanup=True
### Run inference for subj3
##printf "Calling test.py for model trained on all but subj3...\n"
##python scripts/inference.py \
##    --single_denoiser=0 \
##    --set_dir="data/subj3" \
##    --train_data="data/subj4/train" \
##    --result_dir="psnr_results/subj3_results" \
##    --model_dir_all_noise="psnr_noise_estimated_models/AllButsubj3Trained/MyDnCNN_all_noise" \
##    --model_dir_low_noise="psnr_noise_estimated_models/AllButsubj3Trained/MyDnCNN_low_noise" \
##    --model_dir_medium_noise="psnr_noise_estimated_models/AllButsubj3Trained/MyDnCNN_medium_noise" \
##    --model_dir_high_noise="psnr_noise_estimated_models/AllButsubj3Trained/MyDnCNN_high_noise"
##############
#
#### SUBJ4 ###
#printf "Training AllButsubj4Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj5/train" --result_dir="psnr_noise_estimated_models/AllButsubj4Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj5/train" --result_dir="psnr_noise_estimated_models/AllButsubj4Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj5/train" --train_data="data/subj6/train" \
#    --val_data="data/subj5/train" --result_dir="psnr_noise_estimated_models/AllButsubj4Trained" --is_cleanup=True
### Run inference for subj4
##printf "Calling test.py for model trained on all but subj4...\n"
##python scripts/inference.py \
##    --single_denoiser=0 \
##    --set_dir="data/subj4" \
##    --train_data="data/subj5/train" \
##    --result_dir="psnr_results/subj4_results" \
##    --model_dir_all_noise="psnr_noise_estimated_models/AllButsubj4Trained/MyDnCNN_all_noise" \
##    --model_dir_low_noise="psnr_noise_estimated_models/AllButsubj4Trained/MyDnCNN_low_noise" \
##    --model_dir_medium_noise="psnr_noise_estimated_models/AllButsubj4Trained/MyDnCNN_medium_noise" \
##    --model_dir_high_noise="psnr_noise_estimated_models/AllButsubj4Trained/MyDnCNN_high_noise"
##############
#
#### SUBJ5 ###
#printf "Training AllButsubj5Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj5Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj5Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj6/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj5Trained" --is_cleanup=True
### Run inference for subj5
##printf "Calling test.py for model trained on all but subj5...\n"
##python scripts/inference.py \
##    --single_denoiser=0 \
##    --set_dir="data/subj5" \
##    --train_data="data/subj6/train" \
##    --result_dir="psnr_results/subj5_results" \
##    --model_dir_all_noise="psnr_noise_estimated_models/AllButsubj5Trained/MyDnCNN_all_noise" \
##    --model_dir_low_noise="psnr_noise_estimated_models/AllButsubj5Trained/MyDnCNN_low_noise" \
##    --model_dir_medium_noise="psnr_noise_estimated_models/AllButsubj5Trained/MyDnCNN_medium_noise" \
##    --model_dir_high_noise="psnr_noise_estimated_models/AllButsubj5Trained/MyDnCNN_high_noise"
##############
#
#
#### SUBJ6 ###
#printf "Training AllButsubj6Trained model...\n"
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj5/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj6Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj5/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj6Trained" --is_cleanup=True
#python scripts/train.py \
#    --train_data="data/subj1/train" --train_data="data/subj2/train" --train_data="data/subj3/train" \
#    --train_data="data/subj4/train" --train_data="data/subj5/train" \
#    --val_data="data/subj4/train" --result_dir="psnr_noise_estimated_models/AllButsubj6Trained" --is_cleanup=True
### Run inference for subj6
##printf "Calling test.py for model trained on all but subj6...\n"
##python scripts/inference.py \
##    --single_denoiser=0 \
##    --set_dir="data/subj6" \
##    --train_data="data/subj4/train" \
##    --result_dir="psnr_results/subj6_results" \
##    --model_dir_all_noise="psnr_noise_estimated_models/AllButsubj6Trained/MyDnCNN_all_noise" \
##    --model_dir_low_noise="psnr_noise_estimated_models/AllButsubj6Trained/MyDnCNN_low_noise" \
##    --model_dir_medium_noise="psnr_noise_estimated_models/AllButsubj6Trained/MyDnCNN_medium_noise" \
##    --model_dir_high_noise="psnr_noise_estimated_models/AllButsubj6Trained/MyDnCNN_high_noise"



