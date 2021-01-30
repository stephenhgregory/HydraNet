#!/usr/bin/env bash
# Used to run inference on all subj1-subj6

#### SUBJ1 ###
## Train AllButsubj1
#printf "Training AllButsubj1Trained model...\n"
#python scripts/train.py --is_left_middle_right=true --id_portion="low" \
#    --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj2/train" --result_dir="slice_location_models/AllButsubj1Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="middle" \
#    --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj2/train" --result_dir="slice_location_models/AllButsubj1Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="high" \
#    --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj2/train" --result_dir="slice_location_models/AllButsubj1Trained"
## Run inference for subj1
#printf "Calling inference_left_middle_right.py for model trained on all but subj1...\n"
#python scripts/inference_left_middle_right.py \
#    --set_dir="subj1_coregistered_data/subj1" \
#    --result_dir="slice_location_results/subj1_results" \
#    --model_dir_left="slice_location_models/AllButsubj1Trained/MyDnCNN_low_id" \
#    --model_dir_middle="slice_location_models/AllButsubj1Trained/MyDnCNN_middle_id" \
#    --model_dir_right="slice_location_models/AllButsubj1Trained/MyDnCNN_high_id"
#############

### SUBJ2 ###
#printf "Training AllButsubj2Trained model...\n"
#python scripts/train.py --is_left_middle_right=true --id_portion="low" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj3/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj2Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="middle" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj3/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj2Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="high" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj3/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj2Trained"
## Run inference for subj2
#printf "Calling inference_left_middle_right.py for model trained on all but subj2...\n"
#python scripts/inference_left_middle_right.py \
#    --set_dir="subj1_coregistered_data/subj2" \
#    --result_dir="slice_location_results/subj2_results" \
#    --model_dir_left="slice_location_models/AllButsubj2Trained/MyDnCNN_low_id" \
#    --model_dir_middle="slice_location_models/AllButsubj2Trained/MyDnCNN_middle_id" \
#    --model_dir_right="slice_location_models/AllButsubj2Trained/MyDnCNN_high_id"
#############

### SUBJ3 ###
#printf "Training AllButsubj3Trained model...\n"
#python scripts/train.py --is_left_middle_right=true --id_portion="low" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj3Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="middle" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj3Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="high" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj4/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj3Trained"
## Run inference for subj3
#printf "Calling inference_left_middle_right.py for model trained on all but subj3...\n"
#python scripts/inference_left_middle_right.py \
#    --set_dir="subj1_coregistered_data/subj3" \
#    --result_dir="slice_location_results/subj3_results" \
#    --model_dir_left="slice_location_models/AllButsubj3Trained/MyDnCNN_low_id" \
#    --model_dir_middle="slice_location_models/AllButsubj3Trained/MyDnCNN_middle_id" \
#    --model_dir_right="slice_location_models/AllButsubj3Trained/MyDnCNN_high_id"
#############

### SUBJ4 ###
#printf "Training AllButsubj4Trained model...\n"
#python scripts/train.py --is_left_middle_right=true --id_portion="low" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj4Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="middle" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj4Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="high" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj5/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj4Trained"
# Run inference for subj4
printf "Calling inference_left_middle_right.py for model trained on all but subj4...\n"
python scripts/inference_left_middle_right.py \
    --set_dir="subj1_coregistered_data/subj4" \
    --result_dir="left_middle_right_results/subj4_results_all_noise" \
    --model_dir_left="left_middle_right_models/AllButsubj4Trained/MyDnCNN_low_id" \
    --model_dir_middle="models/AllButsubj4Trained/MyDnCNN_all_noise" \
    --model_dir_right="left_middle_right_models/AllButsubj4Trained/MyDnCNN_high_id"
#############

### SUBJ5 ###
#printf "Training AllButsubj5Trained model...\n"
#python scripts/train.py --is_left_middle_right=true --id_portion="low" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj4/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj5Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="middle" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj4/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj5Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="high" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj4/train" --train_data="subj1_coregistered_data/subj6/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj5Trained"
## Run inference for subj5
#printf "Calling inference_left_middle_right.py for model trained on all but subj5...\n"
#python scripts/inference_left_middle_right.py \
#    --set_dir="subj1_coregistered_data/subj5" \
#    --result_dir="slice_location_results/subj5_results" \
#    --model_dir_left="slice_location_models/AllButsubj5Trained/MyDnCNN_low_id" \
#    --model_dir_middle="slice_location_models/AllButsubj5Trained/MyDnCNN_middle_id" \
#    --model_dir_right="slice_location_models/AllButsubj5Trained/MyDnCNN_high_id"
#############


### SUBJ6 ###
#printf "Training AllButsubj6Trained model...\n"
#python scripts/train.py --is_left_middle_right=true --id_portion="low" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj4/train" --train_data="subj1_coregistered_data/subj5/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj6Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="middle" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj4/train" --train_data="subj1_coregistered_data/subj5/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj5Trained"
#python scripts/train.py --is_left_middle_right=true --id_portion="high" \
#    --train_data="subj1_coregistered_data/subj1/train" --train_data="subj1_coregistered_data/subj2/train" --train_data="subj1_coregistered_data/subj3/train" \
#    --train_data="subj1_coregistered_data/subj4/train" --train_data="subj1_coregistered_data/subj5/train" \
#    --val_data="subj1_coregistered_data/subj1/train" --result_dir="slice_location_models/AllButsubj5Trained"
## Run inference for subj6
#printf "Calling inference_left_middle_right.py for model trained on all but subj6...\n"
#python scripts/inference_left_middle_right.py \
#    --set_dir="subj1_coregistered_data/subj6" \
#    --result_dir="slice_location_results/subj6_results" \
#    --model_dir_left="slice_location_models/AllButsubj6Trained/MyDnCNN_low_id" \
#    --model_dir_middle="slice_location_models/AllButsubj6Trained/MyDnCNN_middle_id" \
#    --model_dir_right="slice_location_models/AllButsubj6Trained/MyDnCNN_high_id"
##############


