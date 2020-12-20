import argparse
import sys
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# # This is for running in Pycharm, where the root directory is MyDenoiser, and not MyDenoiser/keras_implementation
# from keras_implementation.utilities import data_generator, logger, image_utils

# This is for running normally, where the root directory is MyDenoiser/keras_implementation
from utilities import data_generator, logger, image_utils

# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', action='append', default=[], type=str, help='path of train data')
args = parser.parse_args()


def show_residual_std_distribution(data_dir: str = args.train_data):
    """
    Function which aggregates a histogram of residual standard deviations and shows it with Matplotlib

    :param data_dir: The directory in which training examples are stored

    :return: None
    """
    print(f'Accessing training data in: {data_dir}')

    # If we are getting train data from one directory...
    if len(data_dir) == 1:
        # Get training examples from data_dir[0] using pair_data_generator
        x_original, y_original = data_generator.pair_data_generator(data_dir[0])

    # Else, if we're getting data from multiple directories...
    elif len(data_dir) > 1:
        # Get training examples from data_dir using pair_data_generator_multiple_data_dirs
        x_original, y_original = data_generator.pair_data_generator_multiple_data_dirs(data_dir)

    # Else, something is wrong - we don't have train data! Exit.
    else:
        sys.exit('ERROR: You didn\'t provide any data directories to train on!')

    # Create lists to store all of the stds for all of the noise residuals
    stds = []

    # Iterate over all of the image patches
    for x_patch, y_patch in zip(x_original, y_original):

        # If the patch is black (i.e. the max px value < 10), just skip this training example
        if np.max(x_patch) < 10:
            continue

        # Get the residual std
        std = data_generator.get_residual_std(clear_patch=x_patch,
                                              blurry_patch=y_patch)

        # Add the patch to the total list of residual stds
        stds.append(std)

    # Convert stds into numpy arrays
    stds = np.array(stds, dtype='float64')

    # Plot the residual standard deviation
    image_utils.plot_standard_deviations(stds)


def show_psnr_distribution(data_dir: str = args.train_data):
    """
    Function which aggregates a histogram of PSNRs and shows it with Matplotlib

    :param data_dir: The directory in which training examples are stored

    :return: None
    """
    print(f'Accessing training data in: {data_dir}')

    # If we are getting train data from one directory...
    if len(data_dir) == 1:
        # Get training examples from data_dir[0] using pair_data_generator
        x_original, y_original = data_generator.pair_data_generator(data_dir[0])
    # Else, if we're getting data from multiple directories...
    elif len(data_dir) > 1:
        # Get training examples from data_dir using pair_data_generator_multiple_data_dirs
        x_original, y_original = data_generator.pair_data_generator_multiple_data_dirs(data_dir)
    # Else, something is wrong - we don't have train data! Exit.
    else:
        sys.exit('ERROR: You didn\'t provide any data directories to train on!')

    # Create lists to store all of the psnrs for all of the patch pairs
    psnrs = []

    # Iterate over all of the image patches
    for x_patch, y_patch in zip(x_original, y_original):

        # If the patch is black (i.e. the max px value < 10), just skip this training example
        if np.max(x_patch) < 10:
            continue

        # Get the PSNR and add to the total list of PSNRs
        psnr = peak_signal_noise_ratio(x_patch, y_patch)
        psnrs.append(psnr)

    # Convert psnrs into numpy arrays
    psnrs = np.array(psnrs, dtype='float64')

    # Plot the residual standard deviation
    image_utils.plot_psnrs(psnrs)


if __name__ == "__main__":

    # Show the residual std distribution
    show_residual_std_distribution(data_dir=args.train_data)

    # Show the PSNR distribution
    show_psnr_distribution(data_dir=args.train_data)
