"""
Contains various testing functions to ensure all modules are working nominally
"""
import cv2
import numpy as np
import time
import image_utils
import data_generator


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


def test_and_plot_residual_stds(data_dir):
    """
    Given a data directory, calculates and plots the residual standard deviations

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


def test_inference_time_train_data_generation(train_data_dir: str) -> None:
    """
    Tests the generation and splitting of train data into 3 noise levels at inference time

    :param train_data_dir: The directory housing training data
    :return:
    """

    print("Beginning getting training patches")
    start_time = time.time()

    # Get our training data to use for determining which denoising network to send each patch through
    training_patches = data_generator.retrieve_train_data(train_data_dir, skip_every=3)

    low_noise_x = training_patches["low_noise"]["x"]
    medium_noise_x = training_patches["medium_noise"]["x"]
    high_noise_x = training_patches["high_noise"]["x"]

    print(f"Done getting training patches! Total time = {time.time() - start_time}")


if __name__ == "__main__":

    test_inference_time_train_data_generation(train_data_dir="../../data/subj6/train")
