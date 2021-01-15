"""Analyzes the effectivess of different patch similarity metrics in esimating the noise level of a patch"""

import argparse
import sys
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
from collections import namedtuple
import cv2
import copy
import pickle
from utilities import data_generator, logger, image_utils
from utilities.image_utils import plot_psnr_comparisons
from typing import Dict, Tuple, List
from tqdm import tqdm

# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--test_data_subj', default='subj1', type=str, help='name of subject used for testing')
parser.add_argument('--reference_data_subj', default='subj2', type=str, help='name of subject used for reference')
parser.add_argument("--lower_psnr_threshold", default=20., type=float, help='lower threshold used to separate patches '
                                                                            'into low, medium, and high noise '
                                                                            'categories')
parser.add_argument("--upper_psnr_threshold", default=35., type=float, help='upper threshold used to separate patches '
                                                                            'into low, medium, and high noise '
                                                                            'categories')
args = parser.parse_args()

# Initialize global variable to keep track of # of patches per noise level
total_patches_per_category = {
    'low': 0,
    'medium': 0,
    'high': 0
}


def compare_to_closest_training_patch_with_statistics(patch: np.ndarray, training_patches_with_statistics: np.ndarray,
                                                      comparison_metric: str = 'ssim') -> Tuple[float, float]:
    """
    Takes an image patch and compares it with all patches in a given set of training patches to find
    the one with max similarity. Returns the similarity between the given image patch and that chosen
    training patch.

    Parameters
    ----------
    patch: The patch to find a closest match to
    training_patches_with_statistics: The set of training patches to compare the input patch with.
        These training patches include the PSNR with their respective clear images.
    comparison_metric: If 'ssim', we find max SSIM, otherwise, if 'psnr', we find max PSNR

    Returns
    -------
    The PSNR or SSIM between the input patch and the closest match in training_patches, as well as the PSNR of the
        chosen closest training patch with respect to its true, clear patch counterpart
    """
    max_score = 0.
    max_patch_psnr = 0.
    score = 0.

    # Make sure that each training patch has an associated statistic
    assert len(training_patches_with_statistics["y"]) == len(training_patches_with_statistics["comparison_metrics"])

    for i in range(len(training_patches_with_statistics["y"])):
        # First, reshape training_patch and patch to get the ssim
        training_patch = training_patches_with_statistics["y"][i].reshape(
            training_patches_with_statistics["y"][i].shape[0], training_patches_with_statistics["y"][i].shape[1])
        patch = patch.reshape(patch.shape[0], patch.shape[1])
        if comparison_metric == 'psnr':
            # Get the PSNR between y_patch and y_low_noise_patch
            score = peak_signal_noise_ratio(patch, training_patch)
        elif comparison_metric == 'ssim':
            # Get the SSIM between y_patch and y_low_noise_patch
            score = structural_similarity(training_patch, patch)
        # Then, reshape the input patch back
        patch = patch.reshape(patch.shape[0], patch.shape[1], 1)
        # If it's greater than the max, update the max
        if score > max_score:
            max_score = score
            max_patch_psnr = training_patches_with_statistics["comparison_metrics"][i]
    return max_score, max_patch_psnr


# def estimate_noise_statistics_by_patches(y: np.ndarray, x: np.ndarray, x_original_mean: float, x_original_std: float,
#                                          y_original_mean: float, y_original_std: float,
#                                          training_patches: Dict = None) -> List[namedtuple('PsnrComparisonTuple', ['true', 'predicted'])]:
def estimate_noise_statistics_by_patches(y: np.ndarray, x: np.ndarray, x_original_mean: float,
                                         x_original_std: float,
                                         y_original_mean: float, y_original_std: float,
                                         training_patches: Dict = None) -> List[Tuple[float, float]]:
    """
    Takes an input image and denoises it using a patch-based approach

    Parameters
    ----------
    y: The input image to denoise
    x: The ground truth, high SNR patch
    x_original_mean: The original mean px value of the image that the patch x is part of, which was used to
                            standardize the image
    x_original_std: The original standard deviation px value of the image that the patch x is part of, which was
                            used to standardize the image
    y_original_mean: The original mean px value of the image that the patch y is part of, which was used to
                            standardize the image
    y_original_std: The original standard deviation px value of the image that the patch y is part of, which was
                            used to standardize the image
    training_patches: A nested dictionary of training patches and their comparison metrics
        (PSNR, SSIM, or residual standard deviation)

    Returns
    -------
    psnr_comparisons: A list of tuples, each of which contains (psnr_x, closest_patch_psnr)
    """
    psnr_comparisons = []

    ''' Just logging
    logger.show_images([("x", x), ("y", y)])
    '''

    # Loop over the indices of y to get 40x40 patches from y
    for i in range(0, len(y[0]), 40):
        for j in range(0, len(y[1]), 40):

            # If the patch does not 'fit' within the dimensions of y, skip this and do not denoise
            if i + 40 > len(y[0]) or j + 40 > len(y[1]):
                continue

            # Get the (40, 40) patches for x and y and reshape the original to be a (40, 40, 1) patch.
            # Then, get a version of the patch with standardization reversed
            y_patch = y[i:i + 40, j:j + 40]
            y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1], 1)
            reversed_y_patch = image_utils.reverse_standardize(y_patch, y_original_mean, y_original_std)
            x_patch = x[i:i + 40, j:j + 40]
            x_patch = x_patch.reshape(x_patch.shape[0], x_patch.shape[1], 1)
            reversed_x_patch = image_utils.reverse_standardize(x_patch, x_original_mean, x_original_std)

            '''Iterate over all of the training patches to get the training patch with the highest 
            SSIM compared to y_patch. Then, use the category of that training image to determine 
            which model to use to denoise this patch'''

            # Get the Max SSIM value and between y_patch and the most similar x in every category, as well as the PSNR
            # of each of those most similar x patches
            low_max_ssim, low_closest_patch_psnr = compare_to_closest_training_patch_with_statistics(reversed_y_patch,
                                                                                                     training_patches[
                                                                                                         "low_noise"],
                                                                                                     comparison_metric='ssim')
            medium_max_ssim, medium_closest_patch_psnr = compare_to_closest_training_patch_with_statistics(
                reversed_y_patch,
                training_patches[
                    "medium_noise"],
                comparison_metric='ssim')
            high_max_ssim, high_closest_patch_psnr = compare_to_closest_training_patch_with_statistics(reversed_y_patch,
                                                                                                       training_patches[
                                                                                                           "high_noise"],
                                                                                                       comparison_metric='ssim')

            # Get the overall max_ssim and PSNR of the closest patch from those above categorical maxes
            max_ssim_category = ''
            closest_patch_psnr = 0.
            max_ssim = max([low_max_ssim, medium_max_ssim, high_max_ssim])
            if max_ssim == high_max_ssim:
                max_ssim_category = 'high'
                closest_patch_psnr = high_closest_patch_psnr
            elif max_ssim == medium_max_ssim:
                max_ssim_category = 'medium'
                closest_patch_psnr = medium_closest_patch_psnr
            elif max_ssim == low_max_ssim:
                max_ssim_category = 'low'
                closest_patch_psnr = low_closest_patch_psnr

            # Get the actual PSNR for x
            psnr_x = peak_signal_noise_ratio(reversed_x_patch, reversed_y_patch)

            ''' Just logging
            logger.show_images([("reversed_x_patch", reversed_x_patch),
                                ("x_patch", x_patch),
                                ("reversed_y_patch", reversed_y_patch)])
            '''

            # Add the closest patch PSNR and actual PSNR to lists to keep track of them
            # PsnrComparisonTuple = namedtuple('PsnrComparisonTuple', ['true', 'predicted'])
            # psnr_comparisons.append(PsnrComparisonTuple(true=psnr_x, predicted=closest_patch_psnr))
            psnr_comparisons.append((psnr_x, closest_patch_psnr))

            # Keep track of total patches called per each category
            total_patches_per_category[max_ssim_category] += 1

    return psnr_comparisons


def pickle_psnr_comparisons(psnr_comparisons: List[Tuple[float, float]], test_data_name: str = None,
                            reference_data_name: str = None, save_dir: str = None) -> None:
    """
    Pickles psnr_comparisons for future retrieval.

    Parameters
    ----------
    psnr_comparisons: Contains the actual PSNR of each patch as well as the predicted PSNR of the patch.
        This predicted PSNR is measured as the actual PSNR of the closest patch.
        The order of the datatype is List[Tuple[actual_psnr, predicted_psnr]]
    test_data_name: Name of data subject used as test data
    reference_data_name: Name of data subject used as reference
    save_dir: Directory in which to save plot

    Returns
    -------
    None
    """
    with open(os.path.join(save_dir, f'{test_data_name}test_{reference_data_name}ref_psnr_comparisons'), 'wb') as pickle_file:
        pickle.dump(psnr_comparisons, pickle_file)


def main():
    reference_data = f"/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/subj1_coregistered_data/{args.reference_data_subj}/train"
    test_data = f"/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/subj1_coregistered_data/{args.test_data_subj}/train"
    lower_psnr_threshold = 20.
    upper_psnr_threshold = 35.

    # Get our training data to use for determining which denoising network to send each patch through
    training_patches = data_generator.retrieve_train_data(reference_data,
                                                          low_noise_threshold=lower_psnr_threshold,
                                                          high_noise_threshold=upper_psnr_threshold,
                                                          skip_every=3,
                                                          patch_size=40,
                                                          stride=20, scales=[1])

    psnr_comparisons = []

    # Iterate over all of the test images from test_data
    image_names = os.listdir(os.path.join(test_data, 'CoregisteredBlurryImages'))
    for image_name in tqdm(image_names):
        if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):
            # 1. Load the Clear Image x (as grayscale), and standardize the pixel values, and..
            # 2. Save the original mean and standard deviation of x
            x, x_orig_mean, x_orig_std = image_utils.standardize(imread(os.path.join(test_data,
                                                                                     'ClearImages',
                                                                                     str(image_name))))

            # Load the Coregistered Blurry Image y (as grayscale), and standardize the pixel values, and...
            # 2. Save the original mean and standard deviation of y
            y, y_orig_mean, y_orig_std = image_utils.standardize(imread(os.path.join(test_data,
                                                                                     'CoregisteredBlurryImages',
                                                                                     str(image_name))))

            psnr_comparisons.extend(estimate_noise_statistics_by_patches(y=y, x=x, x_original_mean=x_orig_mean,
                                                                         x_original_std=x_orig_std,
                                                                         y_original_mean=y_orig_mean,
                                                                         y_original_std=y_orig_std,
                                                                         training_patches=training_patches))

    # Save the PSNR comparisons by pickling them to a binary file
    pickle_psnr_comparisons(psnr_comparisons, test_data_name=args.test_data_subj,
                            reference_data_name=args.reference_data_subj,
                            save_dir="/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/resources"
                                     "/psnr_estimation")

    # Plot/save a scatterplot of predicted vs. actual PSNR
    plot_psnr_comparisons(psnr_comparisons, plot_type="scatterplot",
                          test_data_name=args.test_data_subj, reference_data_name=args.reference_data_subj,
                          save_dir="/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/resources"
                                   "/psnr_estimation",
                          show_plot=False)

    # Plot/save a histogram of PSNR prediction error
    plot_psnr_comparisons(psnr_comparisons, plot_type="histogram",
                          test_data_name=args.test_data_subj, reference_data_name=args.reference_data_subj,
                          save_dir="/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/resources"
                                   "/psnr_estimation",
                          show_plot=False)


if __name__ == "__main__":
    main()
