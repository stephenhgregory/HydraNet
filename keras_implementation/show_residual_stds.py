import argparse
import sys
import numpy as np

# # This is for running in Pycharm, where the root directory is MyDenoiser, and not MyDenoiser/keras_implementation
# from keras_implementation.utilities import data_generator, logger, image_utils

# This is for running normally, where the root directory is MyDenoiser/keras_implementation
from utilities import data_generator, logger, image_utils

# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', action='append', default=[], type=str, help='path of train data')
args = parser.parse_args()


def show_noise_distribution(epoch_iter=2000,
                            batch_size=128,
                            data_dir=args.train_data):
    """
    Generator function that yields training data samples from a specified data directory

    :param epoch_iter: The number of iterations per epoch
    :param batch_size: The number of training examples for each training iteration
    :param data_dir: The directory in which training examples are stored

    :return: Shows the distribution of the standard deviation of the residuals between clear and blurry patches
        in the training datagen for given data directory(s)
    """
    # Loop the following indefinitely...
    while True:
        # Set a counter variable
        counter = 0

        # If this is the first iteration...
        if counter == 0:
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


if __name__ == "__main__":
    # Show the noise distribution
    show_noise_distribution(batch_size=args.batch_size, data_dir=args.train_data)
