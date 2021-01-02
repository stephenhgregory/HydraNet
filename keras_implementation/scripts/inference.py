"""
Main script used to run inference with HydraNet and test the results
"""

import argparse
import os
import time
import datetime
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread, imsave
import tensorflow as tf
import cv2
import copy
from typing import List, Tuple, Dict

# This is for running normally, where the root directory is MyDenoiser/keras_implementation
from utilities import image_utils, logger, data_generator, model_functions

# Set Memory Growth to true to fix a small bug in Tensorflow
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except (ValueError, RuntimeError) as err:
    # Invalid device or cannot modify virtual devices once initialized.
    print(f'tf.config.experimental.set_memory_growth(physical_devices[0], True) threw an error: ')
    print(err)
    pass

# Initialize global variable to keep track of # of patches per noise level
total_patches_per_category = {
    'low': 0,
    'medium': 0,
    'high': 0
}


def parse_args():
    """
    Parses Command Line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/subj1', type=str, help='parent directory of test dataset')
    parser.add_argument('--set_names', default=['train'], type=list, help='name of test dataset')
    # parser.add_argument('--model_dir_original', default=os.path.join('models', 'Volume1Trained', 'MyDnCNN'), type=str,
    #                     help='directory of the original, single-network denoising model')
    parser.add_argument('--model_dir_all_noise',
                        default=os.path.join('models', 'subj1Trained', 'MyDnCNN_all_noise'),
                        type=str,
                        help='directory of the all-noise-denoising model')
    parser.add_argument('--model_dir_low_noise',
                        default=os.path.join('models', 'subj1Trained', 'MyDnCNN_low_noise'),
                        type=str,
                        help='directory of the low-noise-denoising model')
    parser.add_argument('--model_dir_medium_noise',
                        default=os.path.join('models', 'subj1Trained', 'MyDnCNN_medium_noise'),
                        type=str,
                        help='directory of the medium-noise-denoising model')
    parser.add_argument('--model_dir_high_noise',
                        default=os.path.join('models', 'subj1Trained', 'MyDnCNN_high_noise'),
                        type=str,
                        help='directory of the high-noise-denoising model')
    parser.add_argument('--result_dir', default='results/subj1Trained_results/', type=str,
                        help='directory of results')
    parser.add_argument('--reanalyze_data', default=False, type=bool, help='True if we want to simply reanalyze '
                                                                           'results that have already been produced '
                                                                           'and saved')
    parser.add_argument('--train_data', default='data/subj1/train', type=str, help='path of train data')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 for yes or 0 for no')
    parser.add_argument('--single_denoiser', default=1, type=int, help='Use a single denoiser for all noise ranges, '
                                                                       '1 for yes or 0 for no')
    return parser.parse_args()


# TODO: Delete this function
# def to_tensor(image):
#     """ Converts an input image (numpy array) into a tensor """
#
#     if image.ndim == 2:
#         return image[np.newaxis, ..., np.newaxis]
#     elif image.ndim == 3:
#         return np.moveaxis(image, 2, 0)[..., np.newaxis]

# TODO: Delete this function
# def from_tensor(img):
#     """ Converts an image tensor into an image (numpy array) """
#     return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


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


def denoise_image_by_patches(y: np.ndarray, file_name: str, set_name: str, original_mean: float, original_std: float,
                             y_original_mean: float, y_original_std: float, save_patches: bool = True,
                             single_denoiser: bool = False, model_dict: Dict = None,
                             training_patches: Dict = None) -> np.ndarray:
    """
    Takes an input image and denoises it using a patch-based approach

    :param y: The input image to denoise
    :param file_name: The name of the file
    :param set_name: The name of the set containing our test data
    :param original_mean: The original mean px value of the image that the patch x is part of, which was used to
                            standardize the image
    :param original_std: The original standard deviation px value of the image that the patch x is part of, which was
                            used to standardize the image
    :param y_original_mean: The original mean px value of the image that the patch y is part of, which was used to
                            standardize the image
    :param y_original_std: The original standard deviation px value of the image that the patch y is part of, which was
                            used to standardize the image
    :param save_patches: True if we wish to save the individual patches
    :param single_denoiser: True if we wish to denoise patches using only a single denoiser
    :param model_dict: A dictionary of all the TF models used to denoise image patches
    :param training_patches: A nested dictionary of training patches and their residual stds

    :return: x_pred: A denoised image as a numpy array
    :rtype: numpy array
    """

    # Set the save directory name
    save_dir_name = os.path.join(args.result_dir, set_name, file_name + '_patches')

    # First, create a denoised x_pred to INITIALLY be a deep copy of y. Then we will modify x_pred in place
    x_pred = copy.deepcopy(y)

    # Loop over the indices of y to get 40x40 patches from y
    for i in range(0, len(y[0]), 40):
        for j in range(0, len(y[1]), 40):

            # If the patch does not 'fit' within the dimensions of y, skip this and do not denoise
            if i + 40 > len(y[0]) or j + 40 > len(y[1]):
                continue

            # Get the (40, 40) patch, make a copy as a tensor, and then reshape the original to be a (40, 40, 1) patch
            y_patch = y[i:i + 40, j:j + 40]
            y_patch_tensor = image_utils.to_tensor(y_patch)
            y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1], 1)

            '''Iterate over all of the training patches to get the training patch with the highest 
            SSIM compared to y_patch. Then, use the category of that training image to determine 
            which model to use to denoise this patch'''

            # If we wish to use a single denoiser, skip the standard deviation retrieval and parsing into categories,
            # and just denoise each patch
            if single_denoiser:
                print('Calling all-noise model!')
                # Inference with model_low_noise (Denoise y_patch_tensor to get x_patch_pred)
                x_patch_pred_tensor = model_dict["all"].predict(y_patch_tensor)

                # Convert the denoised patch from a tensor to an image (numpy array)
                x_patch_pred = image_utils.from_tensor(x_patch_pred_tensor)

                # Replace the patch in x with the new denoised patch
                x_pred[i:i + 40, j:j + 40] = x_patch_pred

                if save_patches:
                    # Reverse the standardization of x
                    x_patch_pred = image_utils.reverse_standardize(x_patch_pred, original_mean, original_std)

                    # Save the denoised patch
                    image_utils.save_image(x=x_patch_pred,
                                           save_dir_name=save_dir_name,
                                           save_file_name=file_name + '_i-' + str(i) + '_j-' + str(j) + '.png')

                # Finally, skip to the next loop
                continue

            # Get the Max SSIM value between y_patch and the most similar x in every category
            reversed_y_patch = image_utils.reverse_standardize(y_patch, y_original_mean, y_original_std)
            low_max_ssim = compare_to_closest_training_patch(reversed_y_patch, training_patches["low_noise"]["y"],
                                                             comparison_metric='ssim')
            medium_max_ssim = compare_to_closest_training_patch(reversed_y_patch, training_patches["medium_noise"]["y"],
                                                                comparison_metric='ssim')
            high_max_ssim = compare_to_closest_training_patch(reversed_y_patch, training_patches["high_noise"]["y"],
                                                              comparison_metric='ssim')

            # Get the overall max_ssim from those above categorical maxes
            max_ssim_category = ''
            max_ssim = max([low_max_ssim, medium_max_ssim, high_max_ssim])
            if max_ssim == high_max_ssim:
                max_ssim_category = 'high'
            elif max_ssim == medium_max_ssim:
                max_ssim_category = 'medium'
            elif max_ssim == low_max_ssim:
                max_ssim_category = 'low'

            # Keep track of total patches called per each category
            total_patches_per_category[max_ssim_category] += 1
            print(f'Calling {max_ssim_category}-noise model!')

            # Inference with model_low_noise (Denoise y_patch_tensor to get x_patch_pred)
            x_patch_pred_tensor = model_dict[max_ssim_category].predict(y_patch_tensor)

            # Convert the denoised patch from a tensor to an image (numpy array)
            x_patch_pred = image_utils.from_tensor(x_patch_pred_tensor)

            # Replace the patch in x with the new denoised patch
            x_pred[i:i + 40, j:j + 40] = x_patch_pred

            if save_patches:
                # Reverse the standardization of x
                x_patch_pred = image_utils.reverse_standardize(x_patch_pred, original_mean, original_std)

                # Save the denoised patch
                image_utils.save_image(x=x_patch_pred,
                                       save_dir_name=save_dir_name,
                                       save_file_name=file_name + '_i-' + str(i) + '_j-' + str(j) + '.png')

    '''Just logging
    logger.show_images([("y", y), ("x_pred", x_pred)])
    '''

    return x_pred


def compare_to_closest_training_patch(patch: np.ndarray, training_patches: np.ndarray,
                                      comparison_metric: str = 'ssim') -> float:
    """
    Takes an image patch and compares it with all patches in a given set of training patches to find
    the one with max similarity. Returns the similarity between the given image patch and that chosen
    training patch.

    :param patch: The patch to find a closest match to
    :param training_patches: The set of training patches to compare the input patch with
    :param comparison_metric: If 'ssim', we find max SSIM, otherwise, if 'psnr', we find max PSNR
    :return: The PSNR or SSIM between the input patch and the closest match in training_patches
    """
    max_score = 0
    for training_patch in training_patches:
        # First, reshape training_patch and patch to get the ssim
        training_patch = training_patch.reshape(training_patch.shape[0], training_patch.shape[1])
        patch = patch.reshape(patch.shape[0], patch.shape[1])
        if comparison_metric == 'psnr':
            # Get the PSNR between y_patch and y_low_noise_patch
            score = peak_signal_noise_ratio(patch, training_patch)
        elif comparison_metric == 'ssim':
            # Get the SSIM between y_patch and y_low_noise_patch
            score = structural_similarity(training_patch, patch)
        # Then, reshape the input patch back
        patch = patch.reshape(patch.shape[0], patch.shape[1], 1)
        # If it's greater than the max, update the max
        if score > max_score:
            max_score = score
    return max_score


def main(args):
    """The main function of the program"""

    # Get the latest epoch numbers
    # latest_epoch_original = model_functions.findLastCheckpoint(save_dir=args.model_dir_original)
    latest_epoch_all_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_all_noise)
    latest_epoch_low_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_low_noise)
    latest_epoch_medium_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_medium_noise)
    latest_epoch_high_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_high_noise)

    # Create dictionaries to store models and training patches
    model_dict = {}
    training_patches = {}

    # If we are denoising with a single denoiser...
    if args.single_denoiser:
        # Load our single all-noise denoising model
        model_dict["all"] = load_model(os.path.join(args.model_dir_all_noise,
                                                    'model_%03d.hdf5' % latest_epoch_all_noise),
                                       compile=False)
        log(f'Loaded single all-noise model: '
            f'{os.path.join(args.model_dir_all_noise, "model_%03d.hdf5" % latest_epoch_all_noise)}. ')

    # Otherwise...
    else:
        # Load our 3 denoising models
        # model_dict["original"] = load_model(
        #     os.path.join(args.model_dir_original, 'model_%03d.hdf5' % latest_epoch_original),
        #     compile=False)
        ''' TODO: Uncomment this
        model_dict["all"] = load_model(
            os.path.join(args.model_dir_all_noise, 'model_%03d.hdf5' % latest_epoch_all_noise),
            compile=False)
        '''
        model_dict["low"] = load_model(
            os.path.join(args.model_dir_low_noise, 'model_%03d.hdf5' % latest_epoch_low_noise),
            compile=False)
        model_dict["medium"] = load_model(
            os.path.join(args.model_dir_medium_noise, 'model_%03d.hdf5' % latest_epoch_medium_noise),
            compile=False)
        model_dict["high"] = load_model(
            os.path.join(args.model_dir_high_noise, 'model_%03d.hdf5' % latest_epoch_high_noise),
            compile=False)
        log(f'Loaded all 3 trained models: '
            f'{os.path.join(args.model_dir_low_noise, "model_%03d.hdf5" % latest_epoch_low_noise)}, '
            f'{os.path.join(args.model_dir_medium_noise, "model_%03d.hdf5" % latest_epoch_medium_noise)}, and '
            f'{os.path.join(args.model_dir_high_noise, "model_%03d.hdf5" % latest_epoch_high_noise)}')

    if not args.single_denoiser:
        # Get our training data to use for determining which denoising network to send each patch through
        training_patches = data_generator.retrieve_train_data(args.train_data, low_noise_threshold=20.0,
                                                              high_noise_threshold=40.0, skip_every=3, patch_size=40,
                                                              stride=20, scales=[1])

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

                # Skip this example if the result already exists
                if os.path.exists(os.path.join(args.result_dir, set_name, image_name)):
                    continue

                # Get the image name minus the file extension
                image_name_no_extension, _ = os.path.splitext(image_name)

                # 1. Load the Clear Image x (as grayscale), and standardize the pixel values, and..
                # 2. Save the original mean and standard deviation of x
                x, x_orig_mean, x_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         str(set_name),
                                                                                         'ClearImages',
                                                                                         str(image_name)), 0))

                # Load the Coregistered Blurry Image y (as grayscale), and standardize the pixel values, and...
                # 2. Save the original mean and standard deviation of y
                y, y_orig_mean, y_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         str(set_name),
                                                                                         'CoregisteredBlurryImages',
                                                                                         str(image_name)), 0))

                # Start a timer
                start_time = time.time()

                # Denoise the image
                x_pred = denoise_image_by_patches(y=y, file_name=str(image_name_no_extension), set_name=set_name,
                                                  original_mean=x_orig_mean, original_std=x_orig_std,
                                                  y_original_mean=y_orig_mean, y_original_std=y_orig_std,
                                                  save_patches=False, single_denoiser=args.single_denoiser,
                                                  model_dict=model_dict, training_patches=training_patches)

                # Record the inference time
                print('%10s : %10s : %2.4f second' % (set_name, image_name, time.time() - start_time))

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

                # Reverse the standardization of x and x_pred (we actually don't need y at this point, only for logging)
                x = image_utils.reverse_standardize(x, original_mean=x_orig_mean, original_std=x_orig_std)
                x_pred = image_utils.reverse_standardize(x_pred, original_mean=x_orig_mean, original_std=x_orig_std)
                # y = image_utils.reverse_standardize(y, original_mean=y_orig_mean, original_std=y_orig_std)

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

        # Get the average PSNR and SSIM and add into their respective lists
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        # If we want to save the result
        if args.save_result:
            # Save the result to <result_dir>/<set_name>/results.txt
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_name, 'results.txt'))

        # Log the average PSNR and SSIM to the Terminal
        log('Dataset: {0:10s} \n  Average PSNR = {1:2.2f}dB, Average SSIM = {2:1.4f}'.format(set_name, psnr_avg,
                                                                                             ssim_avg))


def reanalyze_denoised_images(set_dir: str, set_names: List[str], result_dir: str, analyze_denoised_data: bool = True,
                              save_results: bool = True) -> Tuple[float, float]:
    """
    Analyzes the denoised data to get SSIM and PSNR values compared to the clean data.
    Also applies masking to remove artifacts from patch denoising

    :param set_dir: The top-level parent directory containing all testing datasets
    :param set_names: A list of test data directories, each of which is a separate dataset
    :param result_dir: The result directtory at which the denoised patches are stored
    :param analyze_denoised_data: True if we wish to analyze denoised images, and
        False if we wish to analyze blurry images instead
    :param save_results: True if we wish to save our results after masking and reanalyzing

    :return: Average PSNR and Average SSIM
        (TODO: Make this a generator because multiple set names will break this function!)
    """

    # Set up whether we are comparing clear data with denoised or blurry data
    comparison_image_type = "denoised_image" if analyze_denoised_data else "blurry_image"

    # TODO: Make this a generator because multiple set names will break this function!
    psnr_avg = 0.0
    ssim_avg = 0.0

    # For each dataset that we wish to test on...
    for set_name in set_names:

        # Create a List of Peak Signal-To-Noise ratios (PSNRs) and Structural Similarities (SSIMs)
        psnrs = []
        ssims = []

        # Iterate over each image in the set
        for image_name in os.listdir(os.path.join(set_dir, set_name, 'CoregisteredBlurryImages')):
            if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):
                # Make sure that we have a matching Clear image, Mask, and comparison image
                assert (os.path.exists(os.path.join(set_dir, set_name, 'ClearImages', image_name)))
                assert (os.path.exists(os.path.join(set_dir, set_name, 'Masks', image_name)))
                if analyze_denoised_data:
                    # assert (os.path.exists(os.path.join(result_dir, set_name, image_name))) # TODO: UNCOMMENT THIS!!!
                    pass
                else:
                    assert (os.path.exists(os.path.join(set_dir, set_name, 'CoregisteredBlurryImages', image_name)))

                # TODO: DELETE THIS!!!!!!
                if not os.path.exists(os.path.join(result_dir, set_name, image_name)):
                    continue
                #########################

                # Load the images
                mask_image = imread(os.path.join(set_dir, set_name, 'Masks', image_name), 0)
                clear_image = imread(os.path.join(set_dir, set_name, 'ClearImages', image_name), 0)
                if analyze_denoised_data:
                    comparison_image = imread(os.path.join(result_dir, set_name, image_name), 0)
                else:
                    comparison_image = imread(os.path.join(set_dir, set_name, 'CoregisteredBlurryImages', image_name),
                                              0)

                ''' Just logging
                logger.show_images([("mask_image",mask_image),
                                    ("clear_image", clear_image),
                                    (comparison_image_type, comparison_image),
                                    ("denoised_image", denoised_image)])
                '''

                # Apply the mask to the clear image AND the comparison (blurry or denoised) image
                comparison_image = comparison_image * (mask_image // 255)
                clear_image = clear_image * (mask_image // 255)

                # Save the denoised image back after applying the mask
                if save_results:
                    cv2.imwrite(filename=os.path.join(args.result_dir, set_name, image_name), img=comparison_image)

                # Get the PSNR and SSIM between clear_image and denoised_image
                psnr = peak_signal_noise_ratio(clear_image, comparison_image)
                ssim = structural_similarity(clear_image, comparison_image, multichannel=True)

                # Add the psnr and ssim to the psnrs and ssim lists, respectively
                if psnr > 0:
                    psnrs.append(psnr)
                ssims.append(ssim)

                ''' Just logging
                logger.show_images([("mask_image",mask_image),
                                    ("clear_image", clear_image),
                                    (comparison_image_type, comparison_image),
                                    ("denoised_image", denoised_image)])
                '''

        # Get the average PSNR and SSIM
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)

        # Log the average PSNR and SSIM to the Terminal
        log('Dataset: {0:10s} \n  Average PSNR = {1:2.2f}dB, Average SSIM = {2:1.4f}'.format(set_name, psnr_avg,
                                                                                             ssim_avg))

    return psnr_avg, ssim_avg


def log_statistics(log_file_path: str, psnr_avg: float, ssim_avg: float):
    """Prints and logs final statistics from inference run"""
    print(f'total low-noise patches: {total_patches_per_category["low"]}')
    print(f'total medium-noise patches: {total_patches_per_category["medium"]}')
    print(f'total high-noise patches: {total_patches_per_category["high"]}')
    with open(log_file_path, 'w') as file:
        file.write(f'Average PSNR = {psnr_avg:2.2f}dB, Average SSIM = {ssim_avg:1.4f}\n')
        file.write(f'total low-noise patches: {total_patches_per_category["low"]}\n')
        file.write(f'total medium-noise patches: {total_patches_per_category["medium"]}\n')
        file.write(f'total high-noise patches: {total_patches_per_category["high"]}\n')


if __name__ == '__main__':

    # Get command-line arguments
    args = parse_args()

    # If the result directory doesn't exist already, just create it
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Run denoising
    if not args.reanalyze_data:
        main(args)

    # Run post-processing (masking) and analysis of results
    psnr_avg, ssim_avg = reanalyze_denoised_images(args.set_dir, args.set_names, args.result_dir,
                                                   save_results=args.save_result)

    log_statistics(log_file_path=os.path.join(args.result_dir, 'log.txt'), psnr_avg=psnr_avg, ssim_avg=ssim_avg)
