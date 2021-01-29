#!/usr/bin/env bash
# Used to run DnCNN denoising (inference) on all subj1-subj6

#### SUBJ1 ###
## Run inference for subj1
#printf "Calling test.py for model trained on all but subj1...\n"
#python scripts/inference.py \
#    --set_dir="data/subj1" \
#    --result_dir="bm3d_results/subj1" \
#    --skip_patch_denoise=1 \
#    --reanalyze_data=1
##############

### SUBJ3 ###
# Run inference for subj3
printf "Calling test.py for model trained on all but subj3...\n"
python scripts/inference.py \
    --set_dir="data/subj3" \
    --result_dir="bm3d_results/subj3" \
    --skip_patch_denoise=1 \
    --reanalyze_data=1
#############
