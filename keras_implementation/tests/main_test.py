import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import keras_implementation.utilities.logger as logger
import keras_implementation.utilities.image_utils as image_utils
import keras_implementation.train as train


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


def main():
    """
    Main method used to call test functions

    :return: None
    """

    test_image_standardization()


if __name__ == "__main__":
    main()
