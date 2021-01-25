#!/usr/bin/env bash
# Used to run ONLY cleanup denoising (inference) on all subj1-subj6

### SUBJ1 ###
## Run inference for subj1 (Residual std noise estimated)
#printf "Calling test.py for model trained on all but subj1...\n"
#python scripts/inference.py \
#    --set_dir="data/subj1" \
#    --result_dir="results/subj1_results" --cleanup_result_dir="results/subj1_cleanup_results" \
#    --model_dir_cleanup="models/AllButsubj1Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
## Run inference for subj1 (PSNR estimated)
#printf "Calling test.py for model trained on all but subj1...\n"
#python scripts/inference.py \
#    --set_dir="data/subj1" \
#    --result_dir="psnr_results/subj1_results" --cleanup_result_dir="psnr_results/subj1_cleanup_results" \
#    --model_dir_cleanup="psnr_noise_estimated_models/AllButsubj1Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
# Run inference for subj1 (single denoiser)
printf "Calling test.py for model trained on all but subj1...\n"
python scripts/inference.py \
    --set_dir="data/subj1" \
    --result_dir="results/subj1_results_single_denoiser" --cleanup_result_dir="results/subj1_single_denoiser_cleanup_results" \
    --model_dir_cleanup="models/AllButsubj1Trained/single_denoiser/MyDnCNN_cleanup" \
    --skip_patch_denoise=1 --cleanup_denoise=1
#############

### SUBJ2 ###
## Run inference for subj2 (Residual std noise estimated)
#printf "Calling test.py for model trained on all but subj2...\n"
#python scripts/inference.py \
#    --set_dir="data/subj2" \
#    --result_dir="results/subj2_results" --cleanup_result_dir="results/subj2_cleanup_results" \
#    --model_dir_cleanup="models/AllButsubj2Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
## Run inference for subj2 (PSNR estimated)
#printf "Calling test.py for model trained on all but subj2...\n"
#python scripts/inference.py \
#    --set_dir="data/subj2" \
#    --result_dir="psnr_results/subj2_results" --cleanup_result_dir="psnr_results/subj2_cleanup_results" \
#    --model_dir_cleanup="psnr_noise_estimated_models/AllButsubj2Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
# Run inference for subj2 (PSNR estimated)
printf "Calling test.py for model trained on all but subj2...\n"
python scripts/inference.py \
    --set_dir="data/subj2" \
    --result_dir="results/subj2_results_single_denoiser" --cleanup_result_dir="results/subj2_single_denoiser_cleanup_results" \
    --model_dir_cleanup="models/AllButsubj2Trained/single_denoiser/MyDnCNN_cleanup" \
    --skip_patch_denoise=1 --cleanup_denoise=1
#############

### SUBJ3 ###
## Run inference for subj3 (Residual std noise estimated)
#printf "Calling test.py for model trained on all but subj3...\n"
#python scripts/inference.py \
#    --set_dir="data/subj3" \
#    --result_dir="results/subj3_results" --cleanup_result_dir="results/subj3_cleanup_results" \
#    --model_dir_cleanup="models/AllButsubj3Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
## Run inference for subj3 (PSNR estimated)
#printf "Calling test.py for model trained on all but subj3...\n"
#python scripts/inference.py \
#    --set_dir="data/subj3" \
#    --result_dir="psnr_results/subj3_results" --cleanup_result_dir="psnr_results/subj3_cleanup_results" \
#    --model_dir_cleanup="psnr_noise_estimated_models/AllButsubj3Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
# Run inference for subj3 (PSNR estimated)
printf "Calling test.py for model trained on all but subj3...\n"
python scripts/inference.py \
    --set_dir="data/subj3" \
    --result_dir="results/subj3_results_single_denoiser" --cleanup_result_dir="results/subj3_single_denoiser_cleanup_results" \
    --model_dir_cleanup="models/AllButsubj3Trained/single_denoiser/MyDnCNN_cleanup" \
    --skip_patch_denoise=1 --cleanup_denoise=1
#############

### SUBJ4 ###
## Run inference for subj4 (Residual std noise estimated)
#printf "Calling test.py for model trained on all but subj4...\n"
#python scripts/inference.py \
#    --set_dir="data/subj4" \
#    --result_dir="results/subj4_results" --cleanup_result_dir="results/subj4_cleanup_results" \
#    --model_dir_cleanup="models/AllButsubj4Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
## Run inference for subj4 (PSNR estimated)
#printf "Calling test.py for model trained on all but subj4...\n"
#python scripts/inference.py \
#    --set_dir="data/subj4" \
#    --result_dir="psnr_results/subj4_results" --cleanup_result_dir="psnr_results/subj4_cleanup_results" \
#    --model_dir_cleanup="psnr_noise_estimated_models/AllButsubj4Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
# Run inference for subj4 (PSNR estimated)
printf "Calling test.py for model trained on all but subj4...\n"
python scripts/inference.py \
    --set_dir="data/subj4" \
    --result_dir="results/subj4_results_single_denoiser" --cleanup_result_dir="results/subj4_single_denoiser_cleanup_results" \
    --model_dir_cleanup="models/AllButsubj4Trained/single_denoiser/MyDnCNN_cleanup" \
    --skip_patch_denoise=1 --cleanup_denoise=1
#############

### SUBJ5 ###
## Run inference for subj5 (Residual std noise estimated)
#printf "Calling test.py for model trained on all but subj5...\n"
#python scripts/inference.py \
#    --set_dir="data/subj5" \
#    --result_dir="results/subj5_results" --cleanup_result_dir="results/subj5_cleanup_results" \
#    --model_dir_cleanup="models/AllButsubj5Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
## Run inference for subj5 (PSNR estimated)
#printf "Calling test.py for model trained on all but subj5...\n"
#python scripts/inference.py \
#    --set_dir="data/subj5" \
#    --result_dir="psnr_results/subj5_results" --cleanup_result_dir="psnr_results/subj5_cleanup_results" \
#    --model_dir_cleanup="psnr_noise_estimated_models/AllButsubj5Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
# Run inference for subj5 (PSNR estimated)
printf "Calling test.py for model trained on all but subj5...\n"
python scripts/inference.py \
    --set_dir="data/subj5" \
    --result_dir="results/subj5_results_single_denoiser" --cleanup_result_dir="results/subj5_single_denoiser_cleanup_results" \
    --model_dir_cleanup="models/AllButsubj5Trained/single_denoiser/MyDnCNN_cleanup" \
    --skip_patch_denoise=1 --cleanup_denoise=1
#############

### SUBJ6 ###
## Run inference for subj6 (Residual std noise estimated)
#printf "Calling test.py for model trained on all but subj6...\n"
#python scripts/inference.py \
#    --set_dir="data/subj6" \
#    --result_dir="results/subj6_results" --cleanup_result_dir="results/subj6_cleanup_results" \
#    --model_dir_cleanup="models/AllButsubj6Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
## Run inference for subj6 (PSNR estimated)
#printf "Calling test.py for model trained on all but subj6...\n"
#python scripts/inference.py \
#    --set_dir="data/subj6" \
#    --result_dir="psnr_results/subj6_results" --cleanup_result_dir="psnr_results/subj6_cleanup_results" \
#    --model_dir_cleanup="psnr_noise_estimated_models/AllButsubj6Trained/MyDnCNN_cleanup" \
#    --skip_patch_denoise=1 --cleanup_denoise=1
# Run inference for subj6 (PSNR estimated)
printf "Calling test.py for model trained on all but subj6...\n"
python scripts/inference.py \
    --set_dir="data/subj6" \
    --result_dir="results/subj6_results_single_denoiser" --cleanup_result_dir="results/subj6_single_denoiser_cleanup_results" \
    --model_dir_cleanup="models/AllButsubj6Trained/single_denoiser/MyDnCNN_cleanup" \
    --skip_patch_denoise=1 --cleanup_denoise=1
#############

