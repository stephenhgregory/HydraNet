"""
Contains various testing functions to ensure all modules are working nominally
"""
import cv2
import numpy as np
import copy
import time
import image_utils
import logger
import os
import data_generator
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread, imsave


def test_image_standardization():
    """
    Tests image standardization functions

    :return: None
    """
    # Open an image with opencv2
    file_name = '/home/ubuntu/PycharmProjects/MyDenoiser/sample_image_folder/pug_photo.jpg'
    img = cv2.imread(file_name, 0)

    # Standardize the image
    standardized_img = image_utils.standardize(img)

    # Recover original image from standardized image
    recovered_img = image_utils.reverse_standardize(standardized_img, img.mean(), img.std())

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    cv2.imshow("Standardized Image", standardized_img)
    cv2.waitKey(0)

    cv2.imshow("Recovered (should match Original) Image", recovered_img)
    cv2.waitKey(0)

    # Destroy the 3 windows we just created
    cv2.destroyWindow('Original Image')
    cv2.destroyWindow('Standardized Image')
    cv2.destroyWindow('Recovered (should match Original) Image')


def test_and_plot_residual_stds(data_dir):
    """
    Given a data directory, calculates and plots the residual standard deviations

    :param data_dir: The directory of the training data
    :type data_dir: str

    :return: None
    """

    # Get training examples from data_dir using data_generator
    x_original, y_original = data_generator.pair_data_generator(data_dir)

    # Iterate over x_original and y_original and get stds
    stds = []
    for x_patch, y_patch in zip(x_original, y_original):
        if np.max(x_patch) < 10:
            continue
        x_patch = x_patch.reshape(x_patch.shape[0], x_patch.shape[1])
        y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1])
        std = data_generator.get_residual_std(clear_patch=x_patch,
                                              blurry_patch=y_patch)
        stds.append(std)
    stds = np.array(stds, dtype='float64')
    stds = stds.reshape(stds.shape[0], 1)

    # Plot the standard deviations
    image_utils.plot_standard_deviations(stds)


def test_and_plot_psnrs(data_dir: str, data_dir_name: str):
    """
    Given a data directory, calculates and plots the residual standard deviations

    :param data_dir: The directory of the training data
    :type data_dir: str

    :return: None
    """

    # Get training examples from data_dir using data_generator
    x_original, y_original = data_generator.pair_data_generator(data_dir, patch_size=40, stride=20, scales=None)

    # Iterate over y_original and get psnrs
    psnrs = []
    concatenated_patches = []
    for x_patch, y_patch in zip(x_original, y_original):
        if np.max(x_patch) < 10:
            continue
        x_patch = x_patch.reshape(x_patch.shape[0], x_patch.shape[1])
        y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1])
        psnr = peak_signal_noise_ratio(x_patch, y_patch)
        psnrs.append(psnr)
        concatenated_patches.append((x_patch, y_patch, psnr))

    # Sort the patches by PSNR
    concatenated_patches = sorted(concatenated_patches, key=lambda x: x[2])

    psnrs = np.array(psnrs, dtype='float64')
    psnrs = psnrs.reshape(psnrs.shape[0], 1)

    # Plot the standard deviations
    image_utils.plot_psnrs(psnrs, data_dir_name=data_dir_name)


def test_inference_time_train_data_generation(train_data_dir: str) -> None:
    """
    Tests the generation and splitting of train data into 3 noise levels at inference time

    :param train_data_dir: The directory housing training data
    :return:
    """

    print("Beginning getting training patches")
    start_time = time.time()

    # Get our training data to use for determining which denoising network to send each patch through
    training_patches = data_generator.retrieve_train_data(train_data_dir, low_noise_threshold=0.04,
                                                          high_noise_threshold=0.15, skip_every=3, patch_size=40,
                                                          stride=20, scales=None)

    low_noise_x = training_patches["low_noise"]["x"]
    medium_noise_x = training_patches["medium_noise"]["x"]
    high_noise_x = training_patches["high_noise"]["x"]

    print(f"Done getting training patches! Total time = {time.time() - start_time}")


def test_inference_time_denoiser_assignment(train_data_dir: str) -> None:
    """TODO: Add documentation"""
    # Get our training data to use for determining which denoising network to send each patch through
    training_patches = data_generator.retrieve_train_data(train_data_dir, low_noise_threshold=0.04,
                                                          high_noise_threshold=0.15, skip_every=3, patch_size=40,
                                                          stride=20, scales=None)

    for image_name in os.listdir('data/subj6/train/CoregisteredBlurryImages'):
        if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):
            # Get the image name minus the file extension
            image_name_no_extension, _ = os.path.splitext(image_name)

            # 1. Load the Clear Image x (as grayscale), and standardize the pixel values, and..
            # 2. Save the original mean and standard deviation of x
            x, x_orig_mean, x_orig_std = image_utils.standardize(imread(os.path.join('data/subj6/train/ClearImages',
                                                                                         str(image_name)), 0))

            # Load the Coregistered Blurry Image y (as grayscale), and standardize the pixel values, and...
            # 2. Save the original mean and standard deviation of y
            y, y_orig_mean, y_orig_std = image_utils.standardize(imread(os.path.join('data/subj6/train/CoregisteredBlurryImages',
                                                                                         str(image_name)), 0))

            # First, create a denoised x_pred to INITIALLY be a deep copy of y. Then we will modify x_pred in place
            x_pred = copy.deepcopy(y)

            # Loop over the indices of y to get 40x40 patches from y
            for i in range(0, len(y[0]), 40):
                for j in range(0, len(y[1]), 40):

                    # If the patch does not 'fit' within the dimensions of y, skip this and do not denoise
                    if i + 40 > len(y[0]) or j + 40 > len(y[1]):
                        continue

                    # Get the (40, 40) patch, make a copy as a tensor, and then reshapethe original to be a (40, 40, 1) patch
                    y_patch = y[i:i + 40, j:j + 40]
                    y_patch_tensor = image_utils.to_tensor(y_patch)
                    y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1], 1)

                    '''Iterate over all of the training patches to get the training patch with the highest 
                    SSIM compared to y_patch. Then, use the category of that training image to determine 
                    which model to use to denoise this patch'''

                    # Initialize variables to hold the max SSIM for each of the low, medium, and high noise datasets
                    max_ssim = 0
                    max_ssim_category = ''

                    # Iterate over every low_noise patch
                    for y_low_noise_patch in training_patches["low_noise"]["y"]:

                        # First, reshape y_low_noise_patch and y_patch to get the ssim
                        y_low_noise_patch = y_low_noise_patch.reshape(y_low_noise_patch.shape[0],
                                                                      y_low_noise_patch.shape[1])
                        y_patch = y_patch.reshape(y_patch.shape[0],
                                                  y_patch.shape[1])

                        # Get the SSIM between y_patch and y_low_noise_patch
                        ssim = structural_similarity(y_low_noise_patch, y_patch)

                        # Then, reshape y_patch back
                        y_patch = y_patch.reshape(y_patch.shape[0],
                                                  y_patch.shape[1],
                                                  1)

                        # If it's greater than the max, update the max
                        if ssim > max_ssim:
                            max_ssim = ssim
                            max_ssim_category = 'low'

                    # Iterate over every medium_noise patch
                    for y_medium_noise_patch in training_patches["medium_noise"]["y"]:

                        # First, reshape y_medium_noise_patch and y_patch to get the ssim
                        y_medium_noise_patch = y_medium_noise_patch.reshape(y_medium_noise_patch.shape[0],
                                                                            y_medium_noise_patch.shape[1])
                        y_patch = y_patch.reshape(y_patch.shape[0],
                                                  y_patch.shape[1])

                        # Get the SSIM between y_patch and y_medium_noise_patch
                        ssim = structural_similarity(y_medium_noise_patch, y_patch)

                        # Then, reshape y_patch back to where it was
                        y_patch = y_patch.reshape(y_patch.shape[0],
                                                  y_patch.shape[1],
                                                  1)

                        # If it's greater than the max, update the max
                        if ssim > max_ssim:
                            max_ssim = ssim
                            max_ssim_category = 'medium'

                    # Iterate over every high_noise patch
                    for y_high_noise_patch in training_patches["high_noise"]["y"]:

                        # First, reshape y_high_noise_patch and y_patch to get the ssim
                        y_high_noise_patch = y_high_noise_patch.reshape(y_high_noise_patch.shape[0],
                                                                        y_high_noise_patch.shape[1])
                        y_patch = y_patch.reshape(y_patch.shape[0],
                                                  y_patch.shape[1])

                        # Get the SSIM between y_patch and y_high_noise_patch
                        ssim = structural_similarity(y_high_noise_patch, y_patch)

                        # Then, reshape y_patch back to what it was
                        y_patch = y_patch.reshape(y_patch.shape[0],
                                                  y_patch.shape[1],
                                                  1)

                        # If it's greater than the max, update the max
                        if ssim > max_ssim:
                            max_ssim = ssim
                            max_ssim_category = 'high'


if __name__ == "__main__":
    # test_inference_time_train_data_generation(train_data_dir="../../data/subj6/train")
    # test_inference_time_denoiser_assignment(train_data_dir="../../data/subj6/train")
    test_and_plot_psnrs(data_dir="../../data/subj3/train", data_dir_name="subj3/train")
