#!/usr/bin/env bash
# Used to run inference on all subj1-subj6

train_psnr_estimated_cleanup_models () {
  ### SUBJ1 ###
  printf "Training AllButsubj1TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="psnr_results/subj2_results/train" --blurry_data="psnr_results/subj3_results/train" \
    --blurry_data="psnr_results/subj4_results/train" --blurry_data="psnr_results/subj5_results/train" \
    --blurry_data="psnr_results/subj6_results/train" \
    --clear_data="data/subj2/train/ClearImages" --clear_data="data/subj3/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="psnr_results/subj1_results/train" --result_dir="psnr_noise_estimated_models/AllButsubj1Trained" --is_cleanup=True
  #############

  ### SUBJ2 ###
  printf "Training AllButsubj2TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="psnr_results/subj1_results/train" --blurry_data="psnr_results/subj3_results/train" \
    --blurry_data="psnr_results/subj4_results/train" --blurry_data="psnr_results/subj5_results/train" \
    --blurry_data="psnr_results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj3/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="psnr_results/subj1_results/train" --result_dir="psnr_noise_estimated_models/AllButsubj2Trained" --is_cleanup=True
  #############

  ### SUBJ3 ###
  printf "Training AllButsubj3TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="psnr_results/subj1_results/train" --blurry_data="psnr_results/subj2_results/train" \
    --blurry_data="psnr_results/subj4_results/train" --blurry_data="psnr_results/subj5_results/train" \
    --blurry_data="psnr_results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="psnr_results/subj4_results/train" --result_dir="psnr_noise_estimated_models/AllButsubj3Trained" --is_cleanup=True
  #############

  ### SUBJ4 ###
  printf "Training AllButsubj4TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="psnr_results/subj1_results/train" --blurry_data="psnr_results/subj2_results/train" \
    --blurry_data="psnr_results/subj3_results/train" --blurry_data="psnr_results/subj5_results/train" \
    --blurry_data="psnr_results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="psnr_results/subj5_results/train" --result_dir="psnr_noise_estimated_models/AllButsubj4Trained" --is_cleanup=True
  #############

  ### SUBJ5 ###
  printf "Training AllButsubj5TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="psnr_results/subj1_results/train" --blurry_data="psnr_results/subj2_results/train" \
    --blurry_data="psnr_results/subj3_results/train" --blurry_data="psnr_results/subj4_results/train" \
    --blurry_data="psnr_results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj4/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="psnr_results/subj6_results/train" --result_dir="psnr_noise_estimated_models/AllButsubj5Trained" --is_cleanup=True
  #############

  ### SUBJ6 ###
  printf "Training AllButsubj6TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="psnr_results/subj1_results/train" --blurry_data="psnr_results/subj2_results/train" \
    --blurry_data="psnr_results/subj3_results/train" --blurry_data="psnr_results/subj4_results/train" \
    --blurry_data="psnr_results/subj5_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj4/train/ClearImages" \
    --clear_data="data/subj5/train/ClearImages" \
    --val_data="psnr_results/subj1_results/train" --result_dir="psnr_noise_estimated_models/AllButsubj6Trained" --is_cleanup=True
  #############
}

train_residual_std_estimated_cleanup_models () {
  ### SUBJ1 ###
  printf "Training AllButsubj1TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj2_results/train" --blurry_data="results/subj3_results/train" \
    --blurry_data="results/subj4_results/train" --blurry_data="results/subj5_results/train" \
    --blurry_data="results/subj6_results/train" \
    --clear_data="data/subj2/train/ClearImages" --clear_data="data/subj3/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj1_results/train" --result_dir="models/AllButsubj1Trained" --is_cleanup=True
  #############

  ### SUBJ2 ###
  printf "Training AllButsubj2TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results/train" --blurry_data="results/subj3_results/train" \
    --blurry_data="results/subj4_results/train" --blurry_data="results/subj5_results/train" \
    --blurry_data="results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj3/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj1_results/train" --result_dir="models/AllButsubj2Trained" --is_cleanup=True
  #############

  ### SUBJ3 ###
  printf "Training AllButsubj3TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results/train" --blurry_data="results/subj2_results/train" \
    --blurry_data="results/subj4_results/train" --blurry_data="results/subj5_results/train" \
    --blurry_data="results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj4_results/train" --result_dir="models/AllButsubj3Trained" --is_cleanup=True
  #############

  ### SUBJ4 ###
  printf "Training AllButsubj4TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results/train" --blurry_data="results/subj2_results/train" \
    --blurry_data="results/subj3_results/train" --blurry_data="results/subj5_results/train" \
    --blurry_data="results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj5_results/train" --result_dir="models/AllButsubj4Trained" --is_cleanup=True
  #############

  ### SUBJ5 ###
  printf "Training AllButsubj5TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results/train" --blurry_data="results/subj2_results/train" \
    --blurry_data="results/subj3_results/train" --blurry_data="results/subj4_results/train" \
    --blurry_data="results/subj6_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj4/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj6_results/train" --result_dir="models/AllButsubj5Trained" --is_cleanup=True
  #############

  ### SUBJ6 ###
  printf "Training AllButsubj6TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results/train" --blurry_data="results/subj2_results/train" \
    --blurry_data="results/subj3_results/train" --blurry_data="results/subj4_results/train" \
    --blurry_data="results/subj5_results/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj4/train/ClearImages" \
    --clear_data="data/subj5/train/ClearImages" \
    --val_data="results/subj1_results/train" --result_dir="models/AllButsubj6Trained" --is_cleanup=True
  #############
}

train_single_denoiser_cleanup_models () {
  ### SUBJ1 ###
  printf "Training AllButsubj1TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj2_results_single_denoiser/train" --blurry_data="results/subj3_results_single_denoiser/train" \
    --blurry_data="results/subj4_results_single_denoiser/train" --blurry_data="results/subj5_results_single_denoiser/train" \
    --blurry_data="results/subj6_results_single_denoiser/train" \
    --clear_data="data/subj2/train/ClearImages" --clear_data="data/subj3/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj1_results/train" --result_dir="models/AllButsubj1Trained/single_denoiser" --is_cleanup=True
  #############

  ### SUBJ2 ###
  printf "Training AllButsubj2TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results_single_denoiser/train" --blurry_data="results/subj3_results_single_denoiser/train" \
    --blurry_data="results/subj4_results_single_denoiser/train" --blurry_data="results/subj5_results_single_denoiser/train" \
    --blurry_data="results/subj6_results_single_denoiser/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj3/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj1_results/train" --result_dir="models/AllButsubj2Trained/single_denoiser" --is_cleanup=True
  #############

  ### SUBJ3 ###
  printf "Training AllButsubj3TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results_single_denoiser/train" --blurry_data="results/subj2_results_single_denoiser/train" \
    --blurry_data="results/subj4_results_single_denoiser/train" --blurry_data="results/subj5_results_single_denoiser/train" \
    --blurry_data="results/subj6_results_single_denoiser/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj4/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj4_results/train" --result_dir="models/AllButsubj3Trained/single_denoiser" --is_cleanup=True
  #############

  ### SUBJ4 ###
  printf "Training AllButsubj4TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results_single_denoiser/train" --blurry_data="results/subj2_results_single_denoiser/train" \
    --blurry_data="results/subj3_results_single_denoiser/train" --blurry_data="results/subj5_results_single_denoiser/train" \
    --blurry_data="results/subj6_results_single_denoiser/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj5/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj5_results/train" --result_dir="models/AllButsubj4Trained/single_denoiser" --is_cleanup=True
  #############

  ### SUBJ5 ###
  printf "Training AllButsubj5TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results_single_denoiser/train" --blurry_data="results/subj2_results_single_denoiser/train" \
    --blurry_data="results/subj3_results_single_denoiser/train" --blurry_data="results/subj4_results_single_denoiser/train" \
    --blurry_data="results/subj6_results_single_denoiser/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj4/train/ClearImages" \
    --clear_data="data/subj6/train/ClearImages" \
    --val_data="results/subj6_results/train" --result_dir="models/AllButsubj5Trained/single_denoiser" --is_cleanup=True
  #############

  ### SUBJ6 ###
  printf "Training AllButsubj6TrainedCleanup model...\n"
  python scripts/train.py \
    --blurry_data="results/subj1_results_single_denoiser/train" --blurry_data="results/subj2_results_single_denoiser/train" \
    --blurry_data="results/subj3_results_single_denoiser/train" --blurry_data="results/subj4_results_single_denoiser/train" \
    --blurry_data="results/subj5_results_single_denoiser/train" \
    --clear_data="data/subj1/train/ClearImages" --clear_data="data/subj2/train/ClearImages" \
    --clear_data="data/subj3/train/ClearImages" --clear_data="data/subj4/train/ClearImages" \
    --clear_data="data/subj5/train/ClearImages" \
    --val_data="results/subj1_results/train" --result_dir="models/AllButsubj6Trained/single_denoiser" --is_cleanup=True
  #############
}

#train_psnr_estimated_cleanup_models
train_residual_std_estimated_cleanup_models
train_single_denoiser_cleanup_models

