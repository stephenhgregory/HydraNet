#!/usr/bin/env bash
# Used to train myDenoiser

# Find and store the MyDenoiser repo path
MYDENOISER_REPO=$(find ~ -type d -wholename "*/MyDenoiser/keras_implementation" )

# Move into the MyDenoiser repo if it exists, otherwise exit with an error message
if [ "$MYDENOISER_REPO" == "\n" ] || [ -z "$MYDENOISER_REPO" ]
then
  echo "Could not find MyDenoiser repository. Clone that repo and try again!" && exit 1
else
  cd "$MYDENOISER_REPO" && echo "MyDenoiser repo found at $MYDENOISER_REPO"
fi

# Get the model type that the user wishes to train
printf "Which model do you want to retrain?\n"
printf "[1] low-noise\n"
printf "[2] medium-noise\n"
printf "[3] high-noise\n"
printf "[4] all-noise (1 model)\n"
printf "[5] I wish to train low-noise, medium-noise, and high-noise models separately!\n"
printf "[6] I wish to train low-noise, medium-noise, high-noise, AND all-noise models separately!\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Get the image directory that the user wishes to train on
printf "Which train data directory do you want to retrain on?\n"
printf "[1] Volume1\n"
printf "[2] Volume2\n"
printf "[3] subj1\n"
printf "[4] subj2\n"
printf "[5] subj1 AND subj2\n"
printf "\n"
printf "Select a number: "
read -r DATA_NUM

# [1] low-noise model training
if [ "$MODEL_NUM" == 1 ]
then
  NOISE_LEVELS[0]="low"
# [2] medium-noise model training
elif [ "$MODEL_NUM" == 2 ]
then
  NOISE_LEVELS[0]="medium"
# [3] high-noise model training
elif [ "$MODEL_NUM" == 3 ]
then
  NOISE_LEVELS[0]="high"
  echo "Calling train.py for high-noise model..."
  python train.py --noise_level=high
# [4] all-noise model training
elif [ "$MODEL_NUM" == 4 ]
then
  NOISE_LEVELS[0]="all"
# [5] train all 3 noise levels separately (low, medium, and high)
elif [ "$MODEL_NUM" == 5 ]
then
  NOISE_LEVELS[0]="low"
  NOISE_LEVELS[1]="medium"
  NOISE_LEVELS[2]="high"
# [6] train all 3 noise levels separately (low, medium, and high) + all-noise model
elif [ "$MODEL_NUM" == 6 ]
then
  NOISE_LEVELS[0]="all"
  NOISE_LEVELS[1]="low"
  NOISE_LEVELS[2]="medium"
  NOISE_LEVELS[3]="high"
fi

# [1] Using Volume1 for training data
if [ "$DATA_NUM" == 1 ]
then
  for noise_level in "${NOISE_LEVELS[@]}"
  do
    printf "Calling train.py for %s-noise model...\n" "$noise_level"
    python train.py --noise_level="${noise_level}" --train_data="data/Volume1/train" --val_data="data/Volume1/val"
  done
# [2] Using Volume2 for training data
elif [ "$DATA_NUM" == 2 ]
then
  for noise_level in "${NOISE_LEVELS[@]}"
  do
    printf "Calling train.py for %s-noise model...\n" "$noise_level"
    python train.py --noise_level="${noise_level}" --train_data="data/Volume2/train" --val_data="data/Volume2/val"
  done
# [3] Using subj1 for training data
elif [ "$DATA_NUM" == 3 ]
then
  for noise_level in "${NOISE_LEVELS[@]}"
  do
    printf "Calling train.py for %s-noise model...\n" "$noise_level"
    python train.py --noise_level="${noise_level}" --train_data="data/subj1/train" --val_data="data/subj1/val"
  done
# [4] Using subj2 for training data
elif [ "$DATA_NUM" == 4 ]
then
  for noise_level in "${NOISE_LEVELS[@]}"
  do
    printf "Calling train.py for %s-noise model...\n" "$noise_level"
    python train.py --noise_level="${noise_level}" --train_data="data/subj2/train" --val_data="data/subj2/val"
  done
# [5] Using subj1 AND subj2 for training data
elif [ "$DATA_NUM" == 5 ]
then
  for noise_level in "${NOISE_LEVELS[@]}"
  do
    printf "Calling train.py for %s-noise model...\n" "$noise_level"
    python train.py --noise_level="${noise_level}" \
    --train_data="data/subj2/train" --train_data="data/subj1/train" \
    --val_data="data/subj2/val"
  done
fi