#!/usr/bin/env bash
# Used for inference in cross-validation study

# Get the model type that the user wishes to test
printf "Which model?\n"
printf "[1] subj1\n"
printf "[2] subj2\n"
printf "[3] subj3\n"
printf "[4] subj4\n"
printf "[5] subj5\n"
printf "[6] subj6\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Ask whether we're using single denoiser or HydraNet (multiple denoisers)
printf "Single Denoiser?\n[y/n]: "
read -r SINGLE_DENOISER

# Set model to perform inference with and train data
if [ "$MODEL_NUM" == 1 ]
then
  DATA="subj1"
  TRAINED_DIR="AllButsubj1"
  TRAIN_DATA_DIR="subj2"
elif [ "$MODEL_NUM" == 2 ]
then
  DATA="subj2"
  TRAINED_DIR="AllButsubj2"
  TRAIN_DATA_DIR="subj3"
elif [ "$MODEL_NUM" == 3 ]
then
  DATA="subj3"
  TRAINED_DIR="AllButsubj3"
  TRAIN_DATA_DIR="subj4"
elif [ "$MODEL_NUM" == 4 ]
then
  DATA="subj4"
  TRAINED_DIR="AllButsubj4"
  TRAIN_DATA_DIR="subj5"
elif [ "$MODEL_NUM" == 5 ]
then
  DATA="subj5"
  TRAINED_DIR="AllButsubj5"
  TRAIN_DATA_DIR="subj2"
elif [ "$MODEL_NUM" == 6 ]
then
  DATA="subj6"
  TRAINED_DIR="AllButsubj6"
  TRAIN_DATA_DIR="subj1"
fi

# Set single denoiser or HydraNet (multiple denoisers)
if [ "$SINGLE_DENOISER" == "y" ]
then
  ALL_NOISE=1
else
  ALL_NOISE=0
fi

#### PSNR Estimation #########################################################################
## Set the name of the result directory based upon the training directory and whether we are
## training a single, all-noise denoiser
#if [ "$ALL_NOISE" == 1 ]
#then
#  RESULT_DIR="psnr_results/${DATA}_results_single_denoiser"
#else
#  RESULT_DIR="psnr_results/${DATA}_results"
#fi
#
#printf "Calling test.py for model trained on %s...\n" "$TRAINED_DIR"
#python scripts/inference.py \
#    --single_denoiser="${ALL_NOISE}" \
#    --set_dir="data/${DATA}" \
#    --train_data="data/${TRAIN_DATA_DIR}/train" \
#    --result_dir="${RESULT_DIR}" \
#    --model_dir_all_noise="psnr_noise_estimated_models/${TRAINED_DIR}Trained/MyDnCNN_all_noise" \
#    --model_dir_low_noise="psnr_noise_estimated_models/${TRAINED_DIR}Trained/MyDnCNN_low_noise" \
#    --model_dir_medium_noise="psnr_noise_estimated_models/${TRAINED_DIR}Trained/MyDnCNN_medium_noise" \
#    --model_dir_high_noise="psnr_noise_estimated_models/${TRAINED_DIR}Trained/MyDnCNN_high_noise"
##############################################################################################

### Residual Standard Deviation Estimation ###################################################
# Set the name of the result directory based upon the training directory and whether we are
# training a single, all-noise denoiser
if [ "$ALL_NOISE" == 1 ]
then
  RESULT_DIR="results/${DATA}_results_single_denoiser"
else
  RESULT_DIR="results/${DATA}_results"
fi

printf "Calling test.py for model trained on %s...\n" "$TRAINED_DIR"
python scripts/inference.py \
    --single_denoiser="${ALL_NOISE}" \
    --set_dir="data/${DATA}" \
    --train_data="data/${TRAIN_DATA_DIR}/train" \
    --result_dir="${RESULT_DIR}" \
    --model_dir_all_noise="models/${TRAINED_DIR}Trained/MyDnCNN_all_noise" \
    --model_dir_low_noise="models/${TRAINED_DIR}Trained/MyDnCNN_low_noise" \
    --model_dir_medium_noise="models/${TRAINED_DIR}Trained/MyDnCNN_medium_noise" \
    --model_dir_high_noise="models/${TRAINED_DIR}Trained/MyDnCNN_high_noise"
##############################################################################################
