#!usr/bin/env python

import os
import keras_implementation.utilities.logger as logger
from os.path import isfile, join
import numpy as np
import shutil
import errno
from pathlib import Path


def main(root_dir=(join(Path(__file__).resolve().parents[1], 'data'))):
    """
    Main method ran by the program to create and populate train, val, and test
    datasets.

    :param root_dir: The root directory of the image dataset
    :type root_dir: basestring
    :return: None
    """

    # Iterate over each volume in the root data directory
    for folder_name in os.listdir(root_dir):
        create_train_test_val_dirs(join(root_dir, folder_name))
        populate_train_test_val_dirs_nonrandomly(join(root_dir, folder_name))


def create_train_test_val_dirs(root_dir):
    """
    Creates empty directories that will hold train, validation, and test
    splits of image dataset.

    :param root_dir: The root directory under which the train, val, and test sets will live
    :type root_dir: basestring
    :return: None
    """
    try:
        # Create training data directories
        os.makedirs(root_dir + '/train')
        os.makedirs(root_dir + '/train/CoregisteredBlurryImages')
        os.makedirs(root_dir + '/train/ClearImages')

        # Create validation data directories
        os.makedirs(root_dir + '/val')
        os.makedirs(root_dir + '/val/CoregisteredBlurryImages')
        os.makedirs(root_dir + '/val/ClearImages')

        # Create testing data directories
        os.makedirs(root_dir + '/test')
        os.makedirs(root_dir + '/test/CoregisteredBlurryImages')
        os.makedirs(root_dir + '/test/ClearImages')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def populate_train_test_val_dirs_nonrandomly(root_dir, val_ratio=0.15, test_ratio=0.05):
    """
    Populates the train, val, and test folders with the images located in root_dir,
    according to val_ratio  and test_ratio

    :param root_dir: The root directory of the image dataset
    :param val_ratio: The desired ratio of val images to total images
    :param test_ratio: The desired ratio of test images to total images
    :return: None
    """

    ''' Creating partitions of the data after shuffling '''
    # Folder to copy images from
    src = join(root_dir, 'CoregisteredBlurryImages')

    all_file_names = [f for f in os.listdir(src) if isfile(join(src, f))]

    # Select the number of images to skip between validation images
    val_skip_number = len(all_file_names) / (val_ratio * len(all_file_names))

    # Select the number of images to skip between test images
    test_skip_number = len(all_file_names) / (test_ratio * len(all_file_names))

    # Get the list of validation file names, test file names, and train file names
    val_file_names = all_file_names[::int(val_skip_number)]
    test_file_names = [filename for filename in all_file_names[::int(test_skip_number + 1)]
                       if filename not in val_file_names]
    train_file_names = [filename for filename in all_file_names
                        if filename not in val_file_names and filename not in test_file_names]

    # Print the file distribution among the folders
    logger.print_file_distribution(len(all_file_names), len(train_file_names), len(val_file_names), len(test_file_names))

    # Copy-Pasting Images
    for name in train_file_names:
        shutil.copy(join(root_dir, 'CoregisteredBlurryImages', name), root_dir + '/train/CoregisteredBlurryImages')
        shutil.copy(join(root_dir, 'ClearImages', name), root_dir + '/train/ClearImages')
    for name in val_file_names:
        shutil.copy(join(root_dir, 'CoregisteredBlurryImages', name), root_dir + '/val/CoregisteredBlurryImages')
        shutil.copy(join(root_dir, 'ClearImages', name), root_dir + '/val/ClearImages')
    for name in test_file_names:
        shutil.copy(join(root_dir, 'CoregisteredBlurryImages', name), root_dir + '/test/CoregisteredBlurryImages')
        shutil.copy(join(root_dir, 'ClearImages', name), root_dir + '/test/ClearImages')


def populate_train_test_val_dirs_randomly(root_dir=(os.getcwd()), val_ratio=0.15, test_ratio=0.05):
    """
    Populates the train, val, and test folders with the images located in root_dir,
    according to val_ratio  and test_ratio

    :param root_dir: The root directory of the image dataset
    :type root_dir: basestring
    :param val_ratio: The desired ratio of val images to total images
    :type val_ratio: float
    :param test_ratio: The desired ratio of test images to total images
    :type test_ratio: float
    :return: None
    """
    # Creating partitions of the data after shuffling
    # Folder to copy images from
    src = root_dir  # The folder to copy images from

    all_file_names = [f for f in os.listdir(src) if isfile(join(src, f))]

    np.random.shuffle(all_file_names)

    train_file_names, val_file_names, test_file_names = np.split(np.array(all_file_names),
                                                                 [int(len(all_file_names) * (
                                                                         1 - val_ratio + test_ratio)),
                                                                  int(len(all_file_names) * (1 - test_ratio))])
    # Print the file distribution amongst the folders
    logger.print_file_distribution(len(all_file_names), len(train_file_names), len(val_file_names), len(test_file_names))

    print(train_file_names)

    # Copy-Pasting Images
    for name in train_file_names:
        shutil.copy(join(root_dir, 'CoregisteredBlurryImages', name), root_dir + '/train/CoregisteredBlurryImages')
        shutil.copy(join(root_dir, 'ClearImages', name), root_dir + '/train/ClearImages')
    for name in val_file_names:
        shutil.copy(join(root_dir, 'CoregisteredBlurryImages', name), root_dir + '/val/CoregisteredBlurryImages')
        shutil.copy(join(root_dir, 'ClearImages', name), root_dir + '/val/ClearImages')
    for name in test_file_names:
        shutil.copy(join(root_dir, 'CoregisteredBlurryImages', name), root_dir + '/test/CoregisteredBlurryImages')
        shutil.copy(join(root_dir, 'ClearImages', name), root_dir + '/test/ClearImages')


if __name__ == "__main__":

    # Get the root directory by going "up" two directories
    root_dir = Path(__file__).resolve().parents[1]

    # Pass that root directory to the main function
    main(root_dir=join(root_dir, 'data'))
