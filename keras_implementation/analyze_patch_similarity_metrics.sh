#!/usr/bin/env bash
# Used to run scripts/analyze_patch_similarity_metrics.py automatically (for cross validation study)

analyze_test_subj1 () {
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj1 --reference_data_subj=subj2
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj1 --reference_data_subj=subj3
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj1 --reference_data_subj=subj4
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj1 --reference_data_subj=subj5
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj1 --reference_data_subj=subj6
}

analyze_test_subj2 () {
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj2 --reference_data_subj=subj1
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj2 --reference_data_subj=subj3
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj2 --reference_data_subj=subj4
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj2 --reference_data_subj=subj5
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj2 --reference_data_subj=subj6
}

analyze_test_subj3 () {
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj3 --reference_data_subj=subj1
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj3 --reference_data_subj=subj2
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj3 --reference_data_subj=subj4
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj3 --reference_data_subj=subj5
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj3 --reference_data_subj=subj6
}

analyze_test_subj4 () {
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj4 --reference_data_subj=subj1
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj4 --reference_data_subj=subj2
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj4 --reference_data_subj=subj3
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj4 --reference_data_subj=subj5
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj4 --reference_data_subj=subj6
}

analyze_test_subj5 () {
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj5 --reference_data_subj=subj1
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj5 --reference_data_subj=subj2
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj5 --reference_data_subj=subj3
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj5 --reference_data_subj=subj4
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj5 --reference_data_subj=subj6
}

analyze_test_subj6 () {
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj6 --reference_data_subj=subj1
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj6 --reference_data_subj=subj2
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj6 --reference_data_subj=subj3
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj6 --reference_data_subj=subj4
  python scripts/analyze_patch_similarity_metrics.py --test_data_subj=subj6 --reference_data_subj=subj5
}

# Run the functions, analyze the data!
print "Analyzing subjects...\n"
analyze_test_subj1
analyze_test_subj2
analyze_test_subj3
analyze_test_subj4
analyze_test_subj5
analyze_test_subj6
print "All done!\n"