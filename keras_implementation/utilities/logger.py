#!/usr/bin/env python

"""
Contains classes, functions, and methods for logging various information to the console
"""


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