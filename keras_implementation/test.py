"""
This script is used to run inference with MyDenoiser and test the results
"""

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from keras_implementation.utilities import image_utils, logger, data_generator
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread, imsave
import tensorflow as tf
import cv2
import copy

# Set Memory Growth to true to fix a small bug in Tensorflow

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print(f'The following line threw an exception: tf.config.experimental.set_memory_growth(physical_devices[0], True)')
    pass


#############################################################


def parse_args():
    """
    Parses Command Line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Volume1', type=str, help='parent directory of test dataset')
    parser.add_argument('--set_names', default=['val'], type=list, help='name of test dataset')
    parser.add_argument('--model_dir_original', default=os.path.join('models', 'Volume1Trained', 'MyDnCNN'), type=str,
                        help='directory of the original, single-network denoising model')
    parser.add_argument('--model_dir_low_noise', default=os.path.join('models', 'Volume2Trained', 'MyDnCNN_low_noise'),
                        type=str,
                        help='directory of the low-noise-denoising model')
    parser.add_argument('--model_dir_medium_noise',
                        default=os.path.join('models', 'Volume2Trained', 'MyDnCNN_medium_noise'), type=str,


                        help='directory of the medium-noise-denoising model')
    parser.add_argument('--model_dir_high_noise',
                        default=os.path.join('models', 'Volume2Trained', 'MyDnCNN_high_noise'), type=str,
                        help='directory of the high-noise-denoising model')
    parser.add_argument('--model_name_original', default='model_023.hdf5', type=str,
                        help='name of the original. single-network model')
    parser.add_argument('--model_name_low_noise', default='model_025.hdf5', type=str,
                        help='name of the low-noise model')
    parser.add_argument('--model_name_medium_noise', default='model_025.hdf5', type=str,
                        help='name of the medium-noise model')
    parser.add_argument('--model_name_high_noise', default='model_025.hdf5', type=str,
                        help='name of the high-noise model')
    parser.add_argument('--result_dir', default='data/Volume2Trained_originalresults/Volume1', type=str,
                        help='directory of results')
    parser.add_argument('--reanalyze_data', default=True, type=bool, help='True if we want to simply reanalyze '
                                                                          'results that have already been produced '
                                                                          'and saved')
    parser.add_argument('--train_data', default='data/Volume2/train', type=str, help='path of train data')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 for yes or 0 for no')
    return parser.parse_args()


def retrieve_train_data(train_data_dir, low_noise_threshold=0.05, high_noise_threshold=0.3):
    """
    Gets and returns the image patches used during training time

    :param train_data_dir: The root directory of the training data
    :type train_data_dir: str
    :param low_noise_threshold: The lower residual image standard deviation threshold used to determine which data
                                should go to which network
    :type low_noise_threshold: float
    :param high_noise_threshold: The upper residual image standard deviation threshold used to determine which data
                                should go to which network
    :type high_noise_threshold: float

    :return: A tuple of the following:
                1. x_low_noise: the clear patches at a low noise level
                2. y_low_noise: the blurry patches at a low noise level
                3. stds_low_noise: the standard deviation of the residuals at a low noise level
                4. x_medium_noise: the clear patches at a medium noise level
                5. y_medium_noise: the blurry patches at a medium noise level
                6. stds_medium_noise: the standard deviation of the residuals at a medium noise level
                7. x_high_noise: the clear patches at a high noise level
                8. y_high_noise: the blurry patches at a high noise level
                9. stds_high_noise: the standard deviation of the residuals at a high noise level
    """

    print(f'Accessing training data in: {train_data_dir}')

    # Get training examples from data_dir using data_generator
    x, y = data_generator.pair_data_generator(train_data_dir)

    # Create lists to store all of the patches and stds for each noise level category
    x_low_noise = []
    y_low_noise = []
    stds_low_noise = []
    x_medium_noise = []
    y_medium_noise = []
    stds_medium_noise = []
    x_high_noise = []
    y_high_noise = []
    stds_high_noise = []

    # Iterate over all of the image patches
    for x_patch, y_patch in zip(x, y):

        # If the patch is black (i.e. the max px value < 10), just skip this training example
        if np.max(x_patch) < 10:
            continue

        # Get the residual std
        std = data_generator.get_residual_std(clear_patch=x_patch,
                                              blurry_patch=y_patch)

        # Add the patches and their residual stds to their corresponding lists based on noise level
        if std < low_noise_threshold:
            x_low_noise.append(x_patch)
            y_low_noise.append(y_patch)
            stds_low_noise.append(std)
            continue
        elif low_noise_threshold < std < high_noise_threshold:
            x_medium_noise.append(x_patch)
            y_medium_noise.append(y_patch)
            stds_medium_noise.append(std)
            continue
        elif std > high_noise_threshold:
            x_high_noise.append(x_patch)
            y_high_noise.append(y_patch)
            stds_high_noise.append(std)
            continue

    # Convert image patches and stds into numpy arrays
    x_low_noise = np.array(x_low_noise, dtype='uint8')
    y_low_noise = np.array(y_low_noise, dtype='uint8')
    stds_low_noise = np.array(stds_low_noise, dtype='float64')
    x_medium_noise = np.array(x_medium_noise, dtype='uint8')
    y_medium_noise = np.array(y_medium_noise, dtype='uint8')
    stds_medium_noise = np.array(stds_medium_noise, dtype='float64')
    x_high_noise = np.array(x_high_noise, dtype='uint8')
    y_high_noise = np.array(y_high_noise, dtype='uint8')
    stds_high_noise = np.array(stds_high_noise, dtype='float64')

    # Return all of the patches and stds for the 3 categories
    return x_low_noise, y_low_noise, stds_low_noise, \
           x_medium_noise, y_medium_noise, stds_medium_noise, \
           x_high_noise, y_high_noise, stds_high_noise


def to_tensor(image):
    """ Converts an input image (numpy array) into a tensor """

    if image.ndim == 2:
        print('The number image dimensions is 2!')
        return image[np.newaxis, ..., np.newaxis]
    elif image.ndim == 3:
        print('The number of image dimensions is 3!')
        return np.moveaxis(image, 2, 0)[..., np.newaxis]


def from_tensor(img):
    """ Converts an image tensor into an image (numpy array) """

    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


def log(*args, **kwargs):
    """ Generic logger function to print current date and time """

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    """ Saves an image or file to a specific path """

    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    """ Creates a matplotlib plot of an input image x """

    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def save_image(x, save_dir_name, save_file_name, original_mean, original_std):
    """
    Save an image x

    :param x: The image to save
    :type x: numpy array
    :param save_dir_name: The save directory of the image patch
    :type save_dir_name: str
    :param save_file_name: The name of the image patch
    :type save_file_name: str
    :param original_mean: The original mean px value of the image that the patch x is part of, which was used to
                            standardize the image
    :type original_mean: float
    :param original_std: The original standard deviation px valueof the image that the patch x is part of, which was
                            used to standardize the image
    :type original_std: float

    :return: None
    """

    # Reverse the standardization x
    x = image_utils.reverse_standardize(x, original_mean=original_mean, original_std=original_std)

    # If the result directory doesn't exist already, just create it
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    # Save the image
    cv2.imwrite(filename=os.path.join(save_dir_name, save_file_name), img=x)


def denoise_image_by_patches(y, file_name, set_name, model_original, model_low_noise, model_medium_noise,
                             model_high_noise, args, original_mean, original_std, save_patches=True):
    """
    Takes an input image and denoises it using a patch-based approach

    :param y: The input image to denoise
    :type y: numpy array
    :param model_original: The original TF model used to denoise all image patches
    :type model_original: TF Model
    :param file_name: The name of the file
    :type file_name: str
    :param set_name: The name of the set containing our test data
    :type set_name: str
    :param model_low_noise: The TF model used to denoise low-noise image patches
    :type model_low_noise: TF Model
    :param model_medium_noise: The TF model used to denoise medium-noise image patches
    :type model_medium_noise: TF Model
    :param model_high_noise: The TF model used to denoise high-noise image patches
    :type model_high_noise: TF Model
    :param args: The command-line arguments for the file
    :param original_mean: The original mean px value of the image that the patch x is part of, which was used to
                            standardize the image
    :type original_mean: float
    :param original_std: The original standard deviation px valueof the image that the patch x is part of, which was
                            used to standardize the image
    :type original_std: float
    :param save_patches: True if we wish to save the individual patches
    :type save_patches: bool

    :return: x_pred: A denoised image as a numpy array
    :rtype: numpy array
    """

    save_dir_name = os.path.join(args.result_dir, set_name, file_name + '_patches')

    # First, create a denoised x_pred to INITIALLY be a deep copy of y. Then we will modify x_pred in place
    x_pred = copy.deepcopy(y)

    # First, get our training data to use for determining which denoising network to send each patch through
    x_low_noise, y_low_noise, stds_low_noise, x_medium_noise, y_medium_noise, stds_medium_noise, x_high_noise, y_high_noise, stds_high_noise = retrieve_train_data(
        args.train_data)

    # Loop over the indices of y to get 40x40 patches from y
    for i in range(0, len(y[0]), 40):
        for j in range(0, len(y[1]), 40):

            # If the patch does not 'fit' within the dimensions of y, skip this and do not denoise
            if i + 40 > len(y[0]) or j + 40 > len(y[1]):
                continue

            # Get the (40, 40) patch
            y_patch = y[i:i + 40, j:j + 40]

            # Convert y_patch to a tensor
            y_patch_tensor = to_tensor(y_patch)

            # Reshape y to be a (40, 40, 1) patch
            y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1], 1)

            '''Iterate over all of the training patches to get the training patch with the highest 
            SSIM compared to y_patch. Then, use the category of that training image to determine 
            which model to use to denoise this patch'''

            # Initialize variables to hold the max SSIMfor each of the low, medium, and high noise datasets
            max_ssim = 0
            max_ssim_category = ''

            # Iterate over every low_noise patch
            for y_low_noise_patch in y_low_noise:

                # First, reshape y_low_noise_patch and y_patch to get the ssim
                y_low_noise_patch = y_low_noise_patch.reshape(y_low_noise_patch.shape[0],
                                                              y_low_noise_patch.shape[1])
                y_patch = y_patch.reshape(y_patch.shape[0],
                                          y_patch.shape[1])

                # Get the SSIM between y_patch and y_low_noise_patch
                ssim = structural_similarity(y_low_noise_patch, y_patch)

                # Then, reshape y_low_noise_patch and y_patch back
                y_low_noise_patch = y_low_noise_patch.reshape(y_low_noise_patch.shape[0],
                                                              y_low_noise_patch.shape[1],
                                                              1)
                y_patch = y_patch.reshape(y_patch.shape[0],
                                          y_patch.shape[1],
                                          1)

                # If it's greater than the max, update the max
                if ssim > max_ssim:
                    max_ssim = ssim
                    max_ssim_category = 'low'

            # Iterate over every medium_noise patch
            for y_medium_noise_patch in y_medium_noise:

                # First, reshape y_medium_noise_patch and y_patch to get the ssim
                y_medium_noise_patch = y_medium_noise_patch.reshape(y_medium_noise_patch.shape[0],
                                                                    y_medium_noise_patch.shape[1])
                y_patch = y_patch.reshape(y_patch.shape[0],
                                          y_patch.shape[1])

                # Get the SSIM between y_patch and y_medium_noise_patch
                ssim = structural_similarity(y_medium_noise_patch, y_patch)

                # Then, reshape y_medium_noise_patch and y_patch back to where they were
                y_medium_noise_patch = y_medium_noise_patch.reshape(y_medium_noise_patch.shape[0],
                                                                    y_medium_noise_patch.shape[1],
                                                                    1)
                y_patch = y_patch.reshape(y_patch.shape[0],
                                          y_patch.shape[1],
                                          1)

                # If it's greater than the max, update the max
                if ssim > max_ssim:
                    max_ssim = ssim
                    max_ssim_category = 'medium'

            # Iterate over every high_noise patch
            for y_high_noise_patch in y_high_noise:

                # First, reshape y_high_noise_patch and y_patch to get the ssim
                y_high_noise_patch = y_high_noise_patch.reshape(y_high_noise_patch.shape[0],
                                                                y_high_noise_patch.shape[1])
                y_patch = y_patch.reshape(y_patch.shape[0],
                                          y_patch.shape[1])

                # Get the SSIM between y_patch and y_high_noise_patch
                ssim = structural_similarity(y_high_noise_patch, y_patch)

                # Then, reshape y_high_noise_patch and y_patch back
                y_high_noise_patch = y_high_noise_patch.reshape(y_high_noise_patch.shape[0],
                                                                y_high_noise_patch.shape[1],
                                                                1)
                y_patch = y_patch.reshape(y_patch.shape[0],
                                          y_patch.shape[1],
                                          1)

                # If it's greater than the max, update the max
                if ssim > max_ssim:
                    max_ssim = ssim
                    max_ssim_category = 'high'

            # If the max SSIM is in the low_noise image dataset, denoise the image using the low_noise
            # denoising model
            if max_ssim_category == 'low':

                # Inference with model_low_noise (Denoise y_patch_tensor to get x_patch_pred)
                x_patch_pred_tensor = model_original.predict(y_patch_tensor)

                # Convert the denoised patch from a tensor to an image (numpy array)
                x_patch_pred = from_tensor(x_patch_pred_tensor)

                # Replace the patch in x with the new denoised patch
                x_pred[i:i + 40, j:j + 40] = x_patch_pred

                if save_patches:
                    # Save the denoised patch
                    save_image(x=x_patch_pred,
                               save_dir_name=os.path.join(args.result_dir, set_name, file_name + '_patches'),
                               save_file_name=file_name + '_i-' + str(i) + '_j-' + str(j) + '.png',
                               original_mean=original_mean,
                               original_std=original_std)

            # Else, if the max SSIM is in the medium_noise image dataset, denoise the image using the
            # medium_noise denoising model
            elif max_ssim_category == 'medium':

                # Inference with model_medium_noise (Denoise y_patch_tensor to get x_patch_pred)
                x_patch_pred_tensor = model_original.predict(y_patch_tensor)

                # Convert the denoised patch from a tensor to an image (numpy array)
                x_patch_pred = from_tensor(x_patch_pred_tensor)

                # Replace the patch in the image y with the new denoised patch
                x_pred[i:i + 40, j:j + 40] = x_patch_pred

                if save_patches:
                    # Save the denoised patch
                    save_image(x=x_patch_pred,
                               save_dir_name=os.path.join(args.result_dir, set_name, file_name + '_patches'),
                               save_file_name=file_name + '_i-' + str(i) + '_j-' + str(j) + '.png',
                               original_mean=original_mean,
                               original_std=original_std)

            # Else, if the max SSIM is in the high_noise image dataset, denoise the image using the
            # high_noise denoising model
            elif max_ssim_category == 'high':

                # Inference with model_high_noise (Denoise y_patch_tensor to get x_patch_pred)
                x_patch_pred_tensor = model_original.predict(y_patch_tensor)

                # Convert the denoised patch from a tensor to an image (numpy array)
                x_patch_pred = from_tensor(x_patch_pred_tensor)

                # Replace the patch in the image y with the new denoised patch
                x_pred[i:i + 40, j:j + 40] = x_patch_pred

                if save_patches:
                    # Save the denoised patch
                    save_image(x=x_patch_pred,
                               save_dir_name=os.path.join(args.result_dir, set_name, file_name + '_patches'),
                               save_file_name=file_name + '_i-' + str(i) + '_j-' + str(j) + '.png',
                               original_mean=original_mean,
                               original_std=original_std)

    '''Just logging
    logger.show_images([("y", y), ("x_pred", x_pred)])
    '''

    return x_pred


def main(args):
    """The main function of the program"""

    # Compile the command line arguments
    args = parse_args()

    # Then, load our 3 denoising models
    model_original = load_model(os.path.join(args.model_dir_original, args.model_name_original),
                                compile=False)
    model_low_noise = load_model(os.path.join(args.model_dir_low_noise, args.model_name_low_noise),
                                 compile=False)
    model_medium_noise = load_model(os.path.join(args.model_dir_medium_noise, args.model_name_medium_noise),
                                    compile=False)
    model_high_noise = load_model(os.path.join(args.model_dir_high_noise, args.model_name_high_noise),
                                  compile=False)
    log(f'Loaded all 3 trained models: {os.path.join(args.model_dir_low_noise, args.model_name_low_noise)}, '
        f'{os.path.join(args.model_dir_medium_noise, args.model_name_medium_noise)}, and '
        f'{os.path.join(args.model_dir_high_noise, args.model_name_high_noise)}')

    # If the result directory doesn't exist already, just create it
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # For each dataset that we wish to test on...
    for set_name in args.set_names:

        # If the <result directory>/<dataset name> doesn't exist already, just create it
        if not os.path.exists(os.path.join(args.result_dir, set_name)):
            os.mkdir(os.path.join(args.result_dir, set_name))

        # Create a List of Peak Signal-To-Noise ratios (PSNRs) and Structural Similarities (SSIMs)
        psnrs = []
        ssims = []

        for image_name in os.listdir(os.path.join(args.set_dir, set_name, 'CoregisteredBlurryImages')):
            if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):

                # Get the image name minus the file extension
                image_name_no_extension, _ = os.path.splitext(image_name)

                # 1. Load the Clear Image x (as grayscale), and standardize the pixel values, and..
                # 2. Save the original mean and standard deviation of x
                x, x_orig_mean, x_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         set_name,
                                                                                         'ClearImages',
                                                                                         image_name), 0))

                # Load the Coregistered Blurry Image y (as grayscale), and standardize the pixel values, and...
                # 2. Save the original mean and standard deviation of y
                y, y_orig_mean, y_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         set_name,
                                                                                         'CoregisteredBlurryImages',
                                                                                         image_name), 0))

                # Start a timer
                start_time = time.time()

                # Denoise y by calling denoise_image_by_patches, which using the 3 denoising models to denoise each
                # patch of the image separately
                x_pred = denoise_image_by_patches(y, image_name_no_extension, set_name, model_original,
                                                  model_low_noise, model_medium_noise, model_high_noise, args,
                                                  x_orig_mean, x_orig_std, save_patches=False)

                # Record the inference time
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_name, image_name, elapsed_time))

                ''' Just logging 
                # Reverse the standardization
                x_pred_reversed = image_utils.reverse_standardize(x_pred,
                                                                  original_mean=x_orig_mean,
                                                                  original_std=x_orig_std)
                x_reversed = image_utils.reverse_standardize(x,
                                                             original_mean=x_orig_mean,
                                                             original_std=x_orig_std)
                y_reversed = image_utils.reverse_standardize(y,
                                                             original_mean=y_orig_mean,
                                                             original_std=y_orig_std)

                logger.show_images([("x", x),
                                    ("x_reversed", x_reversed),
                                    ("x_pred", x_pred),
                                    ("x_pred_reversed", x_pred_reversed),
                                    ("y", y),
                                    ("y_reversed", y_reversed)])
                '''

                # Reverse the standardization of x, x_pred, and y
                x = image_utils.reverse_standardize(x, original_mean=x_orig_mean, original_std=x_orig_std)
                x_pred = image_utils.reverse_standardize(x_pred, original_mean=x_orig_mean, original_std=x_orig_std)
                y = image_utils.reverse_standardize(y, original_mean=y_orig_mean, original_std=y_orig_std)

                ''' Just logging
                logger.show_images([("x", x),
                                    ("x_pred", x_pred),
                                    ("y", y)])
                '''

                # Get the PSNR and SSIM for x
                psnr_x = peak_signal_noise_ratio(x, x_pred)
                ssim_x = structural_similarity(x, x_pred, multichannel=True)

                # If we want to save the result...
                if args.save_result:
                    ''' Just logging
                    # Show the images
                    logger.show_images([("y", y),
                                        ("x_pred", x_pred)])
                    '''

                    # Then save the denoised image
                    cv2.imwrite(filename=os.path.join(args.result_dir, set_name, image_name), img=x_pred)

                # Add the PSNR and SSIM to the lists of PSNRs and SSIMs, respectively
                psnrs.append(psnr_x)
                ssims.append(ssim_x)

        # Get the average PSNR and SSIM
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)

        # Add the average PSNR and SSIM back into their respective lists
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        # If we want to save the result
        if args.save_result:
            # Save the result to <result_dir>/<set_name>/results.txt
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_name, 'results.txt'))

        # Log the average PSNR and SSIM to the Terminal
        log('Dataset: {0:10s} \n  Average PSNR = {1:2.2f}dB, Average SSIM = {2:1.4f}'.format(set_name, psnr_avg,
                                                                                             ssim_avg))


def reanalyze_data(args, save_results=True):
    """
    Analyzes the already-produced inference results to get SSIM and PSNR values.
    If necessary, may also apply masking to remove artifacts from patch denoising

    :param args: The command-line arguments
    :param save_results: True if we wish to save our results after masking and reanalyzing
    :type save_results: bool

    :return: None
    """

    # For each dataset that we wish to test on...
    for set_name in args.set_names:

        # Create a List of Peak Signal-To-Noise ratios (PSNRs) and Structural Similarities (SSIMs)
        psnrs = []
        ssims = []

        # Iterate over each image in the set
        for image_name in os.listdir(os.path.join(args.set_dir, set_name, 'CoregisteredBlurryImages')):
            if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):
                # Make sure that we have a matching ClearImage, Mask, and Denoised Image
                assert (os.path.exists(os.path.join(args.set_dir, set_name, 'ClearImages', image_name)))
                assert (os.path.exists(os.path.join(args.set_dir, set_name, 'Masks', image_name)))
                assert (os.path.exists(os.path.join(args.result_dir, set_name, image_name)))

                # Load the images and standardize them, saving their mean and std along the way
                clear_image = imread(os.path.join(args.set_dir, set_name, 'ClearImages', image_name), 0)
                mask_image = imread(os.path.join(args.set_dir, set_name, 'Masks', image_name), 0)
                denoised_image = imread(os.path.join(args.result_dir, set_name, image_name), 0)

                ''' Just logging '''
                logger.show_images([("clear_image", clear_image),
                                    ("mask_image", mask_image),
                                    ("denoised_image", denoised_image)])

                # Apply the mask to the denoised image AND the clear image
                denoised_image = denoised_image * (mask_image // 255)
                clear_image = clear_image * (mask_image // 255)

                # Save the denoised image back
                cv2.imwrite(filename=os.path.join(args.result_dir, set_name, image_name), img=denoised_image)

                # Get the PSNR and SSIM between clear_image and denoised_image
                psnr = peak_signal_noise_ratio(clear_image, denoised_image)
                ssim = structural_similarity(clear_image, denoised_image, multichannel=True)

                # Add the psnr and ssim to the psnrs and ssim lists, respectively
                psnrs.append(psnr)
                ssims.append(ssim)

                ''' Just logging '''
                logger.show_images([("clear_image", clear_image),
                                    ("mask_image", mask_image),
                                    ("denoised_image", denoised_image)])

        # Get the average PSNR and SSIM

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)

        # Log the average PSNR and SSIM to the Terminal
        log('Dataset: {0:10s} \n  Average PSNR = {1:2.2f}dB, Average SSIM = {2:1.4f}'.format(set_name, psnr_avg,
                                                                                             ssim_avg))


def analyze_blurry_data(args, save_results=True):
    """
    Analyzes the blurry data to get SSIM and PSNR values compared to the clean data.
    If necessary, may also apply masking to remove artifacts from patch denoising

    :param args: The command-line arguments
    :param save_results: True if we wish to save our results after masking and reanalyzing
    :type save_results: bool

    :return: None
    """

    # For each dataset that we wish to test on...
    for set_name in args.set_names:

        # Create a List of Peak Signal-To-Noise ratios (PSNRs) and Structural Similarities (SSIMs)
        psnrs = []
        ssims = []

        # Iterate over each image in the set
        for image_name in os.listdir(os.path.join(args.set_dir, set_name, 'CoregisteredBlurryImages')):
            if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):
                # Make sure that we have a matching ClearImage, Mask, and Blurry Image
                assert (os.path.exists(os.path.join(args.set_dir, set_name, 'ClearImages', image_name)))
                assert (os.path.exists(os.path.join(args.set_dir, set_name, 'Masks', image_name)))
                assert (os.path.exists(os.path.join(args.set_dir, set_name, 'CoregisteredBlurryImages', image_name)))

                # Load the images and standardize them, saving their mean and std along the way
                clear_image = imread(os.path.join(args.set_dir, set_name, 'ClearImages', image_name), 0)
                mask_image = imread(os.path.join(args.set_dir, set_name, 'Masks', image_name), 0)
                blurry_image = imread(os.path.join(args.set_dir, set_name, 'CoregisteredBlurryImages', image_name), 0)

                ''' Just logging
                logger.show_images([("clear_image", clear_image),
                                    ("mask_image", mask_image),
                                    ("blurry_image", blurry_image)])
                '''

                # Apply the mask to the blurry image AND the clear image
                blurry_image = blurry_image * (mask_image // 255)
                clear_image = clear_image * (mask_image // 255)

                # Get the PSNR and SSIM between clear_image and blurry_image
                psnr = peak_signal_noise_ratio(clear_image, blurry_image)
                ssim = structural_similarity(clear_image, blurry_image, multichannel=True)

                # Add the psnr and ssim to the psnrs and ssim lists, respectively
                psnrs.append(psnr)
                ssims.append(ssim)

                ''' Just logging
                logger.show_images([("clear_image", clear_image),
                                    ("mask_image", mask_image),
                                    ("blurry_image", blurry_image)])
                '''

        # Get the average PSNR and SSIM

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)

        # Log the average PSNR and SSIM to the Terminal
        log('Dataset: {0:10s} \n  Average PSNR = {1:2.2f}dB, Average SSIM = {2:1.4f}'.format(set_name, psnr_avg,
                                                                                             ssim_avg))


if __name__ == '__main__':

    # Get command-line arguments
    args = parse_args()

    if not args.reanalyze_data:
        main(args)

    else:
        reanalyze_data(args, args.save_result)
