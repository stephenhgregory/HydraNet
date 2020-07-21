#!/usr/bin/env python

"""
Contains classes, functions, and methods for logging various information to the console
"""

import datetime
import cv2


def print_file_distribution(num_images_total, num_images_train, num_images_val, num_images_test):
    """ Prints the distribution of image files across train, test, and val datasets """

    print('Total images: ', num_images_total)
    print('Training: ', num_images_train)
    print('Validation: ', num_images_val)
    print('Testing: ', num_images_test)


def show_images(images):
    """
    Use cv2.imshow to show a collection of images

    :param images: A list of tuples, each of which containing an image name and the image itself
    :type images: list of tuples

    :return: None
    """

    for image_name, image in images:
        # Show each image, then wait for a keypress
        cv2.imshow(image_name, image)
        cv2.waitKey(0)

    # Finally, destroy all of the windows
    for image_name, _ in images:
        cv2.destroyWindow(image_name)


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def print_numpy_statistics(x, x_name='array'):
    """ Prints various statistics about an input numpy array x """

    print(f"Shape of {x_name}: {x.shape}")

    print(f"Dimensions of {x_name}: {x.ndim}")

    print(f"Mean of {x_name} (No axis specified): {x.mean()}")

    print(f'Variance of {x_name} (No axis specified): {x.var()}')

    print(f'Standard deviation of {x_name} (No axis specified): {x.std()}')

    print('\n\n\n')
