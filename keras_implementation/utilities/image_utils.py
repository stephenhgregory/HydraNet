"""
This file contains various image augmentation functions
"""

import cv2
import numpy as np
import os


def CLAHE_image_folder(image_dir, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Performs Contrast Limited Adaptive Histogram Equalization on a directory of images

    :param image_dir: The directory containing images to be augmented
    :type image_dir: basestring
    :param clip_limit: The contrast limit of any given tile in the transformation
    :type clip_limit: float
    :param tile_grid_size: The size of each sub-patch that gets normalized
    :type tile_grid_size: tuple of ints

    :return: None
    """

    print(image_dir)

    # Iterate over each file in the image directory
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Get the image
            image = cv2.imread(filename=os.path.join(image_dir, file_name), flags=0)
            # Equalize the image
            equalized_image = CLAHE_single_image(image)
            # Save the image
            print(file_name)
            return
            cv2.imwrite(filename=os.path.join(image_dir, file_name), img=equalized_image)


def CLAHE_single_image(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Performs Contrast Limited Adaptive Histogram Equalization on an input image

    :param image: An numpy array representing an image to be equalized
    :type image: Numpy array
    :param clip_limit: The contrast limit of any given tile in the transformation
    :type clip_limit: float
    :param tile_grid_size: The size of each sub-patch that gets normalized
    :type tile_grid_size: tuple of ints

    :return: An augmented image
    :rtype: Numpy array
    """

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Return the augmented image
    return clahe.apply(image)


def standardize(x):
    """
    Standardizes an input image as a numpy array to have a mean of 0 and standard
    deviation of 1.

    :param x: The input image
    :type x: numpy array

    :return: The standardized image as a numpy array
    :rtype: numpy array
    """

    # Convert x to an array of single-precision floats
    x = x.astype('float32')

    # Get the global mean and standard deviation from x
    x_mean, x_std = (x.mean(), x.std())

    # GLobally standardize the pixels
    return (x - x_mean) / x_std


def reverse_standardize(x, original_mean, original_std):
    """
    Takes an input image x (which has been normalized by taking every px value and
    subtracting the mean, then dividing by the standard deviation), and reverses
    that normalization by taking every px value and multiplying it by the
    standard deviation, then subsequently adding the mean.

    :param x: Normalized input image
    :type x: numpy array
    :param original_mean: The mean of the pixel value distribution before standardization
    :type original_mean: float
    :param original_std: The standard deviation of the pixel value distribution before standardization
    :type original_std: float

    :return: Image x with standardization reversed
    :rtype: numpy array
    """

    # Reverse the normalization
    restored_x = x * original_std + original_mean

    # Convert the values to ints before returning
    return restored_x.astype(np.uint8)


def standardize_and_positive_shift(x):
    """
    Standardizes an input image as a numpy array to have a mean of 0.5 and standard
    deviation of about 0.3. This is done so that the pixel values of the image are
    exclusively non-negative, so that we can visualize the images (We can't visualize an
    image with negative values, most libraries only recognize a scale of [0, 1] or [0, 255]).

    :param x: The input image
    :type x: numpy array

    :return: The standardized image as a numpy array
    :rtype: numpy array
    """

    # Standardize and then positively shift x
    return positive_shift(standardize(x))


def positive_shift(x):
    """
    Positively shifts a standardized input image as a numpy array to have pixel values between 0 and 1.
    This is done so that the pixel values of the image are
    exclusively non-negative, so that we can visualize the images (We can't visualize an
    image with negative values, most libraries only recognize a scale of [0, 1] or [0, 255]).

    :param x: The standardized input image
    :type x: numpy array

    :return: The positively shifted image as a numpy array
    :rtype: numpy array
    """

    # Clip the pixel values from [-1, 1]
    x = np.clip(x, -1.0, 1.0)

    # Shift from [-1, 1] to [0, 1] with 0.5 mean
    x = (x + 1.0) / 2.0

    return x


def hist_match_image_folder(root_dir, blurry_dir_name, clear_dir_name):
    """
    Performs Histogram Equalization to match the histograms of all blurry
    photos in the passed-in image directory to all corresponding/matching clear
    photos in the directory.

    :param root_dir: The directory containing images to be augmented
    :type root_dir: basestring
    :param blurry_dir_name: The directory containing blurry images
    :type blurry_dir_name: basestring
    :param clear_dir_name: The directory containing clear images
    :type clear_dir_name: basestring

    :return: None
    """

    print(f'\nroot_dir: {root_dir}')
    print(f'Augmenting the histograms from {os.path.join(root_dir, blurry_dir_name)} to match {os.path.join(root_dir, clear_dir_name)}')

    # Iterate over each file in the blurry image directory
    for file_name in os.listdir(os.path.join(root_dir, blurry_dir_name)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            print(f'Current file: {os.path.join(root_dir, blurry_dir_name, file_name)}')

            # Get a blurry image from the blurry image folder
            blurry_image = cv2.imread(filename=os.path.join(root_dir, blurry_dir_name, file_name), flags=0)

            # Get a clear image from the (matching) clear image folder
            clear_image = cv2.imread(filename=os.path.join(root_dir, clear_dir_name, file_name), flags=0)

            # Augment the blurry image's histogram to match the clear image's histogram
            blurry_image = hist_match(blurry_image, clear_image)

            # Save the blurry image
            cv2.imwrite(filename=os.path.join(root_dir, blurry_dir_name, file_name), img=blurry_image)


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    # get the original shape of the source images
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
