#!/usr/bin/env bash
# Used to test myDenoiser

# THIS BLOCK ALLOWS ONE TO EFFECTIVELY RUN THIS SCRIPT FROM ANYWHERE ON THE HOST MACHINE
#########################################################################################
## Find and store the MyDenoiser repo path
#MYDENOISER_REPO=$(find ~ -type d -wholename "*/MyDenoiser/keras_implementation" )
#
## Move into the MyDenoiser repo if it exists, otherwise exit with an error message
#if [ "$MYDENOISER_REPO" == "\n" ] || [ -z "$MYDENOISER_REPO" ]
#then
#  echo "Could not find MyDenoiser repository. Clone that repo and try again!" && exit 1
#else
#  cd "$MYDENOISER_REPO" && echo "MyDenoiser repo found at $MYDENOISER_REPO"
#fi
#########################################################################################

# Get the model type that the user wishes to test
printf "Which model do you want to test?\n"
printf "[1] Volume1-trained (all 3 models)\n"
printf "[2] Volume2-trained (all 3 models)\n"
printf "[3] subj1-trained (all 3 models)\n"
printf "[4] subj2-trained (all 3 models)\n"
printf "[5] subj3-trained (all 3 models)\n"
printf "[6] subj4-trained (all 3 models)\n"
printf "[7] subj5-trained (all 3 models)\n"
printf "[8] Volume1-trained (1 all-noise model)\n"
printf "[9] Volume2-trained (1 all-noise model)\n"
printf "[10] subj1-trained (1 all-noise model)\n"
printf "[11] subj2-trained (1 all-noise model)\n"
printf "[12] subj3-trained (1 all-noise model)\n"
printf "[13] subj4-trained (1 all-noise model)\n"
printf "[14] subj5-trained (1 all-noise model)\n"
printf "[15] (subj1 through subj5)-trained (1 all-noise model)\n"
printf "[16] (subj1 through subj5)-trained (all 3 models)\n"
printf "[17] (subj1, subj2, subj3, subj4, subj6)-trained (1 all-noise model)\n"
printf "[18] (subj1, subj2, subj3, subj4, subj6)-trained (all 3 models)\n"
printf "[19] (subj1, subj2, subj3, subj5, subj6)-trained (1 all-noise model)\n"
printf "[20] (subj1, subj2, subj3, subj5, subj6)-trained (all 3 models)\n"
printf "[21] (subj1, subj2, subj4, subj5, subj6)-trained (1 all-noise model)\n"
printf "[22] (subj1, subj2, subj4, subj5, subj6)-trained (all 3 models)\n"
printf "[23] (subj1, subj3, subj4, subj5, subj6)-trained (1 all-noise model)\n"
printf "[24] (subj1, subj3, subj4, subj5, subj6)-trained (all 3 models)\n"
printf "[25] (subj2, subj3, subj4, subj5, subj6)-trained (1 all-noise model)\n"
printf "[26] (subj2, subj3, subj4, subj5, subj6)-trained (all 3 models)\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Get the image directory that the user wishes to test on
printf "Which data directory do you want to retest on?\n"
printf "[1] Volume1\n"
printf "[2] Volume2\n"
printf "[3] subj1\n"
printf "[4] subj2\n"
printf "[5] subj3\n"
printf "[6] subj4\n"
printf "[7] subj5\n"
printf "[8] subj6\n"
printf "\n"
printf "Select a number: "
read -r DATA_NUM

# [1] Volume1-trained (all 3 models)
if [ "$MODEL_NUM" == 1 ]
then
  TRAINED_DIR="Volume1"
  TRAIN_DATA_DIR="Volume1"
  ALL_NOISE=0
# [2] Volume2-trained (all 3 models)
elif [ "$MODEL_NUM" == 2 ]
then
  TRAINED_DIR="Volume2"
  TRAIN_DATA_DIR="Volume2"
  ALL_NOISE=0
# [3] subj1-trained (all 3 models)
elif [ "$MODEL_NUM" == 3 ]
then
  TRAINED_DIR="subj1"
  TRAIN_DATA_DIR="subj1"
  ALL_NOISE=0
# [4] subj2-trained (all 3 models)
elif [ "$MODEL_NUM" == 4 ]
then
  TRAINED_DIR="subj2"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=0
# [5] subj3-trained (all 3 models)
elif [ "$MODEL_NUM" == 5 ]
then
  TRAINED_DIR="subj3"
  TRAIN_DATA_DIR="subj3"
  ALL_NOISE=0
# [6] subj4-trained (all 3 models)
elif [ "$MODEL_NUM" == 6 ]
then
  TRAINED_DIR="subj4"
  TRAIN_DATA_DIR="subj4"
  ALL_NOISE=0
# [7] subj5-trained (all 3 models)
elif [ "$MODEL_NUM" == 7 ]
then
  TRAINED_DIR="subj5"
  TRAIN_DATA_DIR="subj5"
  ALL_NOISE=0
# [8] Volume1-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 8 ]
then
  TRAINED_DIR="Volume1"
  TRAIN_DATA_DIR="Volume1"
  ALL_NOISE=1
# [9] Volume2-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 9 ]
then
  TRAINED_DIR="Volume2"
  TRAIN_DATA_DIR="Volume2"
  ALL_NOISE=1
# [10] subj1-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 10 ]
then
  TRAINED_DIR="subj1"
  TRAIN_DATA_DIR="subj1"
  ALL_NOISE=1
# [11] subj2-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 11 ]
then
  TRAINED_DIR="subj2"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=1
# [12] subj3-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 12 ]
then
  TRAINED_DIR="subj3"
  TRAIN_DATA_DIR="subj3"
  ALL_NOISE=1
# [13] subj4-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 13 ]
then
  TRAINED_DIR="subj4"
  TRAIN_DATA_DIR="subj4"
  ALL_NOISE=1
# [14] subj5-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 14 ]
then
  TRAINED_DIR="subj5"
  TRAIN_DATA_DIR="subj5"
  ALL_NOISE=1
# [15] (subj1 through subj5)-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 15 ]
then
  TRAINED_DIR="subj1subj2subj3subj4Andsubj5"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=1
# [16] (subj1 through subj5)-trained (all 3 models)
elif [ "$MODEL_NUM" == 16 ]
then
  TRAINED_DIR="subj1subj2subj3subj4Andsubj5"
  TRAIN_DATA_DIR="subj1"
  ALL_NOISE=0
# [17] (subj1, subj2, subj3, subj4, subj6)-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 17 ]
then
  TRAINED_DIR="subj1subj2subj3subj4Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=1
# [18] (subj1, subj2, subj3, subj4, subj6)-trained (all 3 models)
elif [ "$MODEL_NUM" == 18 ]
then
  TRAINED_DIR="subj1subj2subj3subj4Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=0
# [19] (subj1, subj2, subj3, subj5, subj6)-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 19 ]
then
  TRAINED_DIR="subj1subj2subj3subj5Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=1
# [20] (subj1, subj2, subj3, subj5, subj6)-trained (all 3 models)
elif [ "$MODEL_NUM" == 20 ]
then
  TRAINED_DIR="subj1subj2subj3subj5Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=0
# [21] (subj1, subj2, subj4, subj5, subj6)-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 21 ]
then
  TRAINED_DIR="subj1subj2subj4subj5Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=1
# [22] (subj1, subj2, subj4, subj5, subj6)-trained (all 3 models)
elif [ "$MODEL_NUM" == 22 ]
then
  TRAINED_DIR="subj1subj2subj4subj5Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=0
# [23] (subj1, subj3, subj4, subj5, subj6)-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 23 ]
then
  TRAINED_DIR="subj1subj3subj4subj5Andsubj6"
  TRAIN_DATA_DIR="subj1"
  ALL_NOISE=1
# [24] (subj1, subj3, subj4, subj5, subj6)-trained (all 3 models)
elif [ "$MODEL_NUM" == 24 ]
then
  TRAINED_DIR="subj1subj3subj4subj5Andsubj6"
  TRAIN_DATA_DIR="subj1"
  ALL_NOISE=0
# [25] (subj2, subj3, subj4, subj5, subj6)-trained (1 all-noise model)
elif [ "$MODEL_NUM" == 25 ]
then
  TRAINED_DIR="subj2subj3subj4subj5Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=1
# [26] (subj2, subj3, subj4, subj5, subj6)-trained (all 3 models)
elif [ "$MODEL_NUM" == 26 ]
then
  TRAINED_DIR="subj2subj3subj4subj5Andsubj6"
  TRAIN_DATA_DIR="subj2"
  ALL_NOISE=0
fi

# [1] Using Volume1 for test data
if [ "$DATA_NUM" == 1 ]
then
  SET_DIR="Volume1"
# [2] Using Volume2 for test data
elif [ "$DATA_NUM" == 2 ]
then
  SET_DIR="Volume2"
# [3] Using subj1 for test data
elif [ "$DATA_NUM" == 3 ]
then
  SET_DIR="subj1"
# [4] Using subj2 for test data
elif [ "$DATA_NUM" == 4 ]
then
  SET_DIR="subj2"
# [5] Using subj3 for test data
elif [ "$DATA_NUM" == 5 ]
then
  SET_DIR="subj3"
# [6] Using subj4 for test data
elif [ "$DATA_NUM" == 6 ]
then
  SET_DIR="subj4"
# [7] Using subj5 for test data
elif [ "$DATA_NUM" == 7 ]
then
  SET_DIR="subj5"
# [8] Using subj6 for test data
elif [ "$DATA_NUM" == 8 ]
then
  SET_DIR="subj6"
fi

# Set the name of the result directory based upon the training directory and whether we are
# training a single, all-noise denoiser
if [ "$ALL_NOISE" == 1 ]
then
  RESULT_DIR="results/${TRAINED_DIR}Trained${SET_DIR}Tested_results_single_denoiser"
else
  RESULT_DIR="results/${TRAINED_DIR}Trained${SET_DIR}Tested_results"
fi

printf "Calling test.py for model trained on %s...\n" "$TRAINED_DIR"
python scripts/inference.py \
    --single_denoiser="${ALL_NOISE}" \
    --set_dir="data/${SET_DIR}" \
    --train_data="data/${TRAIN_DATA_DIR}/train" \
    --result_dir="${RESULT_DIR}" \
    --model_dir_all_noise="models/${TRAINED_DIR}Trained/MyDnCNN_all_noise" \
    --model_dir_low_noise="models/${TRAINED_DIR}Trained/MyDnCNN_low_noise" \
    --model_dir_medium_noise="models/${TRAINED_DIR}Trained/MyDnCNN_medium_noise" \
    --model_dir_high_noise="models/${TRAINED_DIR}Trained/MyDnCNN_high_noise"