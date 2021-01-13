"""
Various image augmentation functions
"""

import cv2
import numpy as np
import SimpleITK
import glob
import os
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Tuple
from collections import namedtuple


def save_image(x, save_dir_name, save_file_name):
    """
    Saves an image x

    :param x: The image to save
    :type x: numpy array
    :param save_dir_name: The save directory of the image patch
    :type save_dir_name: str
    :param save_file_name: The name of the image patch
    :type save_file_name: str

    :return: None
    """

    # If the result directory doesn't exist already, just create it
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)
    # Save the image
    cv2.imwrite(filename=os.path.join(save_dir_name, save_file_name), img=x)


def pngs_to_nii(png_folder_name: str, output_file_name: str) -> None:
    """
    Converts a folder of PNG files into a 3D NIfTI file and saves that NIfTI file to the same folder
    :param png_folder_name: The path to the folder containing PNG files
    :param output_file_name: The name of the output NIfTI (.nii) file

    :return: None
    """

    file_names = sorted(glob.glob(os.path.join(png_folder_name, '*.png')))
    reader = SimpleITK.ImageSeriesReader()
    reader.SetFileNames(file_names)
    vol = reader.Execute()
    SimpleITK.WriteImage(vol, os.path.join(png_folder_name, output_file_name + '.nii.gz'))


def CLAHE_image_folder(image_dir, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Performs Contrast Limited Adaptive Histogram Equalization on a directory of images

    :param image_dir: The directory containing images to be augmented
    :type image_dir: str
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

            ''' Just logging
            logger.show_images([("image", image),
                                ("equalized_image", equalized_image)])
            '''

            # Save the image
            cv2.imwrite(filename=os.path.join(image_dir, file_name), img=equalized_image)


def CLAHE_single_image(image, clip_limit=2.0, tile_grid_size=(8, 8)):
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

    :return: A tuple containing: 1. The standardized image
                                 2. The mean pixel value of the original input
                                 3. The standard deviation pixel value of the original input
    :rtype: numpy array
    """

    # Convert x to an array of single-precision floats
    x = x.astype('float32')

    # Get the global mean and standard deviation from x
    original_mean, original_std = (x.mean(), x.std())

    # Globally standardize the pixels
    if original_std != 0.0:
        standardized_x = (x - original_mean) / original_std
    else:
        standardized_x = (x - original_mean)

    # Return a tuple containing the image, mean, and standard deviation
    return standardized_x, original_mean, original_std


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

    # Clip the values from [0, 255]
    restored_x = np.clip(restored_x, 0., 255.)

    # Convert the values to the proper dtype
    restored_x = restored_x.astype(np.uint8)

    # Convert the values to the proper dtype before returning
    return restored_x


def nlm_denoise_single_image(image: np.ndarray) -> np.ndarray:
    """
    Applies Non-Local Means filtering to denoise an input image x

    Parameters
    ----------
    image: The input image to be denoised

    Returns
    -------
    The denoised image
    """

    # Estimate the sigma (noise) value for the image
    sigma_estimation = np.mean(estimate_sigma(image))

    # Denoise the image
    return denoise_nl_means(image, fast_mode=True, patch_size=5, patch_distance=3)


def nlm_denoise_single_image_name(image_name):
    """
    Applies Non-Local Means filtering to denoise an image located at the path <image_name>

    :param image_name: The path of the image to be denoised
    :type image_name: str

    :return: The denoised image
    """

    # Read the image as grayscale
    image = cv2.imread(image_name, 0)

    # Denoise and return the image
    return nlm_denoise_single_image(image)


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


def hist_match_image_folder(root_dir, blurry_dir_name, clear_dir_name, match_to_clear=True):
    """
    Performs Histogram Equalization to match the histograms of all blurry
    photos in the passed-in image directory to all corresponding/matching clear
    photos in the directory.

    :param root_dir: The directory containing images to be augmented
    :type root_dir: str
    :param blurry_dir_name: The directory containing blurry images
    :type blurry_dir_name: str
    :param clear_dir_name: The directory containing clear images
    :type clear_dir_name: str
    :param match_to_clear: True if you want to alter the blurry image's histogram to match the clear image,
                            False if you want to alter the clear image's histogram to match the blurry image
    :type match_to_clear: bool

    :return: None
    """

    print(f'\nroot_dir: {root_dir}')

    # Iterate over each file in the blurry image directory
    for file_name in os.listdir(os.path.join(root_dir, blurry_dir_name)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            print(f'Current file: {os.path.join(root_dir, blurry_dir_name, file_name)}')

            # Get a blurry image from the blurry image folder
            blurry_image = cv2.imread(filename=os.path.join(root_dir, blurry_dir_name, file_name), flags=0)

            # Get a clear image from the (matching) clear image folder
            clear_image = cv2.imread(filename=os.path.join(root_dir, clear_dir_name, file_name), flags=0)

            # If we want to change the blurry image's histogram to match the clear image...
            if match_to_clear:
                # Augment the blurry image's histogram to match the clear image's histogram
                blurry_image = hist_match(blurry_image, clear_image)

                # Save the blurry image
                cv2.imwrite(filename=os.path.join(root_dir, blurry_dir_name, file_name), img=blurry_image)

            # Else, if we want to change the clear image's histogram to match the blurry image's histogram...
            else:
                # Augment the clear image's histogram to match the blurry image's histogram
                clear_image = hist_match(clear_image, blurry_image)

                # Save the clear image
                cv2.imwrite(filename=os.path.join(root_dir, clear_dir_name, file_name), img=clear_image)


def plot_psnrs(psnrs: np.ndarray, data_dir_name: str, save_dir: str) -> None:
    """
    Takes a numpy array of PSNRs, performs any flattening if needed, and plots them.

    :param psnrs: Collection of psnrs
    :param data_dir_name: The name of the directory from which the PSNRs were obtained
    :param save_dir: The directory in which to save the matplotlib plot

    :return: None
    """

    # Flatten the array so that it is 1-dimensional
    psnrs = psnrs.flatten()

    # seaborn histogram
    sns.distplot(psnrs, hist=True, kde=False,
                 bins=int(20), color='blue',
                 hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title(f'PSNR of {data_dir_name}')
    plt.xlabel('Peak Signal to Noise Ratio (PSNR)')
    plt.ylabel('Patches')

    # Optionally save the plot
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, data_dir_name + '_psnrs.png'))

    # Show the plot
    plt.show()


def plot_standard_deviations(stds):
    """
    Takes a numpy array of standard deviations, performs any flattening if needed, and plots them.

    :param stds: Collection of standard deviations
    :type stds: numpy array

    :return: None
    """

    # Flatten the array so that it is 1-dimensional
    stds = stds.flatten()

    # seaborn histogram
    sns.distplot(stds, hist=True, kde=False,
                 bins=int(20), color='blue',
                 hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title('Standard deviation of residual images')
    plt.xlabel('Residual Standard Deviation')
    plt.ylabel('Patches')
    plt.show()


# def plot_psnr_comparisons(psnr_comparisons: List[namedtuple('PsnrComparisonTuple', ['true', 'predicted'])],
#                           plot_type: str = "scatterplot") -> None:

def plot_psnr_comparisons(psnr_comparisons: List[Tuple[float, float]], plot_type: str = "scatterplot",
                          test_data_name: str = None, reference_data_name: str = None, save_dir: str = None) -> None:
    """
    Plots PSNR estimates for a list of patches compared to the true PSNRs

    Parameters
    ----------
    psnr_comparisons: Contains the actual PSNR of each patch as well as the predicted PSNR of the patch.
        This predicted PSNR is measured as the actual PSNR of the closest patch.
        The order of the datatype is List[Tuple[actual_psnr, predicted_psnr]]
    plot_type: One of ["scatterplot", "histogram"].
        If "scatterplot", we will show a scatterplot of true vs. predicted PSNRs
        If "histogram", we will show a histogram of errors in PSNR distributions
    test_data_name: Name of data subject used as test data
    reference_data_name: Name of data subject used as reference
    save_dir: Directory in which to save plot

    Returns
    -------
    None
    """
    # Extract the relevant lists from psnr_comparisons
    # true_psnrs = [comparison.true for comparison in psnr_comparisons]
    # predicted_psnrs = [comparison.predicted for comparison in psnr_comparisons]
    # psnr_differences = [comparison.true - comparison.predicted for comparison in psnr_comparisons]

    true_psnrs = [comparison[0] for comparison in psnr_comparisons]
    predicted_psnrs = [comparison[1] for comparison in psnr_comparisons]
    psnr_differences = [comparison[0] - comparison[1] for comparison in psnr_comparisons]

    # Convert these lists into numpy arrays
    true_psnrs = np.array(true_psnrs)
    predicted_psnrs = np.array(predicted_psnrs)
    psnr_differences = np.array(psnr_differences)

    # If we wish to show a scatterplot of true vs. predicted PSNRs
    if plot_type == "scatterplot":
        # Seaborn scatterplot
        sns.scatterplot(x=true_psnrs, y=predicted_psnrs)
        # Add labels
        plt.title('True PSNR of training patches and Predicted PSNR (PSNR of closest patch)')
        plt.xlabel(f'True PSNRs ({test_data_name})')
        plt.ylabel(f'Predicted PSNRs ({reference_data_name})')
        # Optionally save the plot
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'{test_data_name}test_{reference_data_name}ref_psnr_estimation.png'))
        # Show the plot
        plt.show()

    # IF we wish to show a histogram of errors in PSNR estimations
    elif plot_type == "histogram":
        # seaborn histogram
        sns.histplot(psnr_differences, hist=True, kde=False, stat="probability",
                     bins=int(20), color='blue',
                     hist_kws={'edgecolor': 'black'})
        # Add labels
        plt.title('Histogram of PSNR Prediction error')
        plt.xlabel('Residual Standard Deviation')
        plt.ylabel('Patches')
        # Optionally save the plot
        # TODO: Implement saving this plot
        # Show the plot
        plt.show()

    pass


def get_residual(clear_image, blurry_image):
    """
    Calculate the residual (difference) between a blurry image and a
    matching clear image.

    :param clear_image: A clear image
    :type clear_image: numpy array
    :param blurry_image: A blurry image matching the above clear_image
    :type blurry_image: numpy array

    :return: The residual difference between the clear image and blurry image
    :rtype: numpy array
    """

    # Convert blurry_image and clear_image into 2 dimensional arrays -- from (x,x,1) to (x,x,)
    blurry_image = blurry_image.reshape(blurry_image.shape[0], blurry_image.shape[1])
    clear_image = clear_image.reshape(clear_image.shape[0], clear_image.shape[1])

    # Throw away the SSIM score and keep the residual between the two images
    (_, residual) = structural_similarity(blurry_image, clear_image, full=True)

    return residual


def get_3d_image_volume(image_dir: str) -> np.ndarray:
    """

    Parameters
    ----------
    image_dir: Directory containing image slices

    Returns
    -------
    All images in image_dir concatenated together into a 3d image volume
    """
    images_and_numbers = []
    # Iterate over the entire list of images
    for i, file_name in enumerate(os.listdir(image_dir)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Open the image
            image = cv2.imread(os.path.join(image_dir, file_name), 0)
            # Get the number of the image
            image_number = list(map(int, re.findall(r'\d+', file_name)))[0]
            # Add a tuple containing the image and its number to the list of images
            images_and_numbers.append((image, image_number))

    # Sort the list of images and numbers by number
    images_and_numbers = sorted(images_and_numbers, key=lambda x: x[1])
    # Only get the images from the images and numbers list
    images = np.array([image for image, number in images_and_numbers])
    return images


def get_3d_residual(clear_image_volume, blurry_image_volume):
    """
    Calculate the residual (difference) between a blurry image volume and a
    matching clear image volume.

    :param clear_image_volume: A clear image volume
    :type clear_image_volume: numpy array
    :param blurry_image_volume: A blurry image volume matching the above clear_image
    :type blurry_image_volume: numpy array

    :return: The residual difference between the clear image and blurry image
    :rtype: numpy array
    """

    # Convert blurry_image and clear_image into 2 dimensional arrays -- from (x,x,x,1) to (x,x,x,)
    blurry_image_volume = blurry_image_volume.reshape(blurry_image_volume.shape[0], blurry_image_volume.shape[1], blurry_image_volume.shape[2])
    clear_image_volume = clear_image_volume.reshape(clear_image_volume.shape[0], clear_image_volume.shape[1], clear_image_volume.shape[2])

    # Throw away the SSIM score and keep the residual between the two images
    (_, residual) = structural_similarity(blurry_image_volume, clear_image_volume, full=True)

    return residual


def hist_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Parameters:
    -----------
    source: Image to transform; the histogram is computed over the flattened
        array
    template: Template image; can have different dimensions to source

    Returns:
    -----------
    matched: The transformed output image
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


def to_tensor(image):
    """ Converts an input image (numpy array) into a tensor """

    if image.ndim == 2:
        return image[np.newaxis, ..., np.newaxis]
    elif image.ndim == 3:
        return np.moveaxis(image, 2, 0)[..., np.newaxis]


def from_tensor(img):
    """ Converts an image tensor into an image (numpy array) """
    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


if __name__ == "__main__":
    image_folder = '/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/data/subj4/ClearImages'
    image_volume = get_3d_image_volume(image_dir=image_folder)
