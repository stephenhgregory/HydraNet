#!/usr/bin/env bash
# Used to create train | validation | test split of input data for model training

# Find and store the MyDenoiser repo path
MYDENOISER_REPO=$(find ~ -type d -wholename "*/MyDenoiser/keras_implementation" )

# Move into the MyDenoiser repo if it exists, otherwise exit with an error message
if [ "$MYDENOISER_REPO" == "\n" ] || [ -z "$MYDENOISER_REPO" ]
then
  echo "Could not find MyDenoiser repository. Clone that repo and try again!" && exit 1
else
  cd "$MYDENOISER_REPO" && echo "MyDenoiser repo found at $MYDENOISER_REPO"
fi

# Run the python script to create the data splits
python utilities/create_train_val_test.py
