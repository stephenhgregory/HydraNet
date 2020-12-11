#!/usr/bin/env bash
# Used to show plot of residual standard deviations in a dataset or group of datasets

# Find and store the MyDenoiser repo path
MYDENOISER_REPO=$(find ~ -type d -wholename "*/MyDenoiser/keras_implementation" )

# Move into the MyDenoiser repo if it exists, otherwise exit with an error message
if [ "$MYDENOISER_REPO" == "\n" ] || [ -z "$MYDENOISER_REPO" ]
then
  echo "Could not find MyDenoiser repository. Clone that repo and try again!" && exit 1
else
  cd "$MYDENOISER_REPO" && echo "MyDenoiser repo found at $MYDENOISER_REPO"
fi

# Get the image directory that the user wishes to inspect
printf "Which train data directory do you want to look at?\n"
printf "[1] Volume1\n"
printf "[2] Volume2\n"
printf "[3] subj1\n"
printf "[4] subj2\n"
printf "[5] subj3\n"
printf "[6] subj4\n"
printf "[7] subj5\n"
printf "[8] subj1, subj2, subj3, subj4 AND subj5\n"
printf "[9] subj1, subj2, subj3, subj4 AND subj6\n"
printf "[10] subj1, subj2, subj3, subj5 AND subj6\n"
printf "[11] subj1, subj2, subj4, subj5 AND subj6\n"
printf "[12] subj1, subj3, subj4, subj5 AND subj6\n"
printf "[13] subj2, subj3, subj4, subj5 AND subj6\n"
printf "\n"
printf "Select a number: "
read -r DATA_NUM

# [1] Evaluating Volume1
if [ "$DATA_NUM" == 1 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/Volume1/train"
# [2] Evaluating Volume2
elif [ "$DATA_NUM" == 2 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/Volume2/train"
# [3] Evaluating subj1
elif [ "$DATA_NUM" == 3 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj1/train"
# [4] Evaluating subj2
elif [ "$DATA_NUM" == 4 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj2/train"
# [5] Evaluating subj3
elif [ "$DATA_NUM" == 5 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj3/train"
# [6] Evaluating subj4
elif [ "$DATA_NUM" == 6 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj4/train"
# [7] Evaluating subj5
elif [ "$DATA_NUM" == 7 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj5/train"
# [8] Evaluating subj1, subj2, subj3, subj4 AND subj5
elif [ "$DATA_NUM" == 8 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj1/train" --train_data="data/subj2/train" \
  --train_data="data/subj3/train" --train_data="data/subj4/train" --train_data="data/subj5/train"
# [9] Evaluating subj1, subj2, subj3, subj4 AND subj6
elif [ "$DATA_NUM" == 9 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj1/train" --train_data="data/subj2/train" \
  --train_data="data/subj3/train" --train_data="data/subj4/train" --train_data="data/subj6/train"
# [10] Evaluating subj1, subj2, subj3, subj5 AND subj6
elif [ "$DATA_NUM" == 10 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj1/train" --train_data="data/subj2/train" \
  --train_data="data/subj3/train" --train_data="data/subj5/train" --train_data="data/subj6/train"
# [11] Evaluating subj1, subj2, subj4, subj5 AND subj6
elif [ "$DATA_NUM" == 11 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj1/train" --train_data="data/subj2/train" \
  --train_data="data/subj4/train" --train_data="data/subj5/train" --train_data="data/subj6/train"
# [12] Evaluating subj1, subj3, subj4, subj5 AND subj6
elif [ "$DATA_NUM" == 12 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj1/train" --train_data="data/subj3/train" \
  --train_data="data/subj4/train" --train_data="data/subj5/train" --train_data="data/subj6/train"
# [13] Evaluating subj2, subj3, subj4, subj5 AND subj6
elif [ "$DATA_NUM" == 13 ]
then
  printf "Calling show_residual_stds.py to evaluate noise residual distribution...\n"
  python show_residual_stds.py --train_data="data/subj2/train" --train_data="data/subj3/train" \
  --train_data="data/subj4/train" --train_data="data/subj5/train" --train_data="data/subj6/train"
fi