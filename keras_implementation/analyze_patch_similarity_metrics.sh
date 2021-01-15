#!/usr/bin/env bash
# Used to run scripts/analyze_patch_similarity_metrics.py automatically (for cross validation study)

### Study for Cross Validation ###
# Test: subj1, Reference: subj2
#python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj1 --reference_data_subj=subj2

# Test: subj2, Reference: subj3
#python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj2 --reference_data_subj=subj3

# Test: subj3, Reference: subj4
python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj3 --reference_data_subj=subj4

# Test: subj4, Reference: subj5
python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj4 --reference_data_subj=subj5

# Test: subj5, Reference: subj6
python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj5 --reference_data_subj=subj6

# Test: subj6, Reference: subj1
python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj6 --reference_data_subj=subj1

print "All done! Bye!\n"