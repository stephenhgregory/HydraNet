from keras_implementation.scripts.utilities import data_generator, image_utils
from os.path import join
import numpy as np

def main(data_dir):
    """
    The main function for this file

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


if __name__ == '__main__':
    main(data_dir=join('../../data', 'Volume1', 'train'))