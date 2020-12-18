#!/usr/bin/env bash
# Used to run inference on all subj1-subj6

## Run inference for subj1
#printf "Calling test.py for model trained on all but subj1...\n"
#python scripts/inference.py \
#    --single_denoiser=0 \
#    --set_dir="data/subj1" \
#    --train_data="data/subj2/train" \
#    --result_dir="results/AllButsubj1Trainedsubj1Tested_results" \
#    --model_dir_all_noise="models/subj2subj3subj4subj5Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_low_noise="models/subj2subj3subj4subj5Andsubj6Trained/MyDnCNN_low_noise" \
#    --model_dir_medium_noise="models/subj2subj3subj4subj5Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_high_noise="models/subj2subj3subj4subj5Andsubj6Trained/MyDnCNN_high_noise"

## Run inference for subj2
#printf "Calling test.py for model trained on all but subj2...\n"
#python scripts/inference.py \
#    --single_denoiser=0 \
#    --set_dir="data/subj2" \
#    --train_data="data/subj3/train" \
#    --result_dir="results/AllButsubj2Trainedsubj2Tested_results" \
#    --model_dir_all_noise="models/subj1subj3subj4subj5Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_low_noise="models/subj1subj3subj4subj5Andsubj6Trained/MyDnCNN_low_noise" \
#    --model_dir_medium_noise="models/subj1subj3subj4subj5Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_high_noise="models/subj1subj3subj4subj5Andsubj6Trained/MyDnCNN_high_noise"

# Run inference for subj3
printf "Calling test.py for model trained on all but subj3...\n"
python scripts/inference.py \
    --single_denoiser=0 \
    --set_dir="data/subj3" \
    --train_data="data/subj4/train" \
    --result_dir="results/AllButsubj3Trainedsubj3Tested_results_no_high_noise_model2" \
    --model_dir_all_noise="models/subj1subj2subj4subj5Andsubj6Trained/MyDnCNN_all_noise" \
    --model_dir_low_noise="models/subj1subj2subj4subj5Andsubj6Trained/MyDnCNN_low_noise" \
    --model_dir_medium_noise="models/subj1subj2subj4subj5Andsubj6Trained/MyDnCNN_all_noise" \
    --model_dir_high_noise="models/subj1subj2subj4subj5Andsubj6Trained/MyDnCNN_all_noise"


## Run inference for subj4
#printf "Calling test.py for model trained on all but subj4...\n"
#python scripts/inference.py \
#    --single_denoiser=0 \
#    --set_dir="data/subj4" \
#    --train_data="data/subj5/train" \
#    --result_dir="results/AllButsubj4Trainedsubj4Tested_results" \
#    --model_dir_all_noise="models/subj1subj2subj3subj5Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_low_noise="models/subj1subj2subj3subj5Andsubj6Trained/MyDnCNN_low_noise" \
#    --model_dir_medium_noise="models/subj1subj2subj3subj5Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_high_noise="models/subj1subj2subj3subj5Andsubj6Trained/MyDnCNN_high_noise"

## Run inference for subj5
#printf "Calling test.py for model trained on all but subj5...\n"
#python scripts/inference.py \
#    --single_denoiser=0 \
#    --set_dir="data/subj5" \
#    --train_data="data/subj6/train" \
#    --result_dir="results/AllButsubj5Trainedsubj5Tested_results" \
#    --model_dir_all_noise="models/subj1subj2subj3subj4Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_low_noise="models/subj1subj2subj3subj4Andsubj6Trained/MyDnCNN_low_noise" \
#    --model_dir_medium_noise="models/subj1subj2subj3subj4Andsubj6Trained/MyDnCNN_all_noise" \
#    --model_dir_high_noise="models/subj1subj2subj3subj4Andsubj6Trained/MyDnCNN_high_noise"

## Run inference for subj6
#printf "Calling test.py for model trained on all but subj6...\n"
#python scripts/inference.py \
#    --single_denoiser=0 \
#    --set_dir="data/subj6" \
#    --train_data="data/subj4/train" \
#    --result_dir="results/AllButsubj6Trainedsubj6Tested_results" \
#    --model_dir_all_noise="models/subj1subj2subj3subj4Andsubj5Trained/MyDnCNN_all_noise" \
#    --model_dir_low_noise="models/subj1subj2subj3subj4Andsubj5Trained/MyDnCNN_low_noise" \
#    --model_dir_medium_noise="models/subj1subj2subj3subj4Andsubj5Trained/MyDnCNN_all_noise" \
#    --model_dir_high_noise="models/subj1subj2subj3subj4Andsubj5Trained/MyDnCNN_high_noise"



















