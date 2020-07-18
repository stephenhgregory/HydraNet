#!/usr/bin/env python

"""
Contains classes, functions, and methods for logging various information to the console
"""

import datetime


def print_file_distribution(num_images_total, num_images_train, num_images_val, num_images_test):
    """
    Prints the distribution of image files across train, test, and val datasets

    :param num_images_total: Total number of images in dataset
    :type num_images_total: int
    :param num_images_train: Number of training images in dataset
    :type num_images_train: int
    :param num_images_val: Number of validation images in dataset
    :type num_images_val: int
    :param num_images_test: Number of test images in dataset
    :type num_images_test: int
    :return: None
    """
    print('Total images: ', num_images_total)
    print('Training: ', num_images_train)
    print('Validation: ', num_images_val)
    print('Testing: ', num_images_test)


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def print_numpy_statistics(x, x_name='array'):
    """
    Prints various statistic about an input nupy array x
    :param x: numpy array that we want statistics about
    :type x: numpy array
    :param x_name: The name of the array
    :type x_name: str

    :return: None
    """

    print(f"Shape of {x_name}: {x.shape}")

    print(f"Dimensions of {x_name}: {x.ndim}")

    print(f"Mean of {x_name} (Along axis 0 by axis 1): {x.mean(axis=(0, 1))}")

    print(f"Mean of {x_name} (No axis specified): {x.mean()}")

    print(f'Variance of {x_name} (Along axis 0 by axis 1): {x.var()}')

    print(f'Standard deviation of {x_name} (Along axis 0 by axis 1): {x.std()}')
