# Run this script to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from keras_implementation.utilities import image_utils, logger
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread, imsave
import tensorflow as tf
import cv2

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
    parser.add_argument('--set_names', default=['train'], type=list, help='name of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('models', 'MyDnCNN'), type=str,
                        help='directory of the model')
    parser.add_argument('--model_name', default='model_023.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='data/results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 for yes or 0 for no')
    return parser.parse_args()


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


def main():
    """The main function of the program"""

    args = parse_args()

    # =============================================================================
    #     # serialize model to JSON
    #     model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
    #     model_json = model.to_json()
    #     with open("model.json", "w") as json_file:
    #         json_file.write(model_json)
    #     # serialize weights to HDF5
    #     model.save_weights("model.h5")
    #     print("Saved model")
    # =============================================================================

    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        print(f'No file exists with the name: {args.model_dir}/{args.model_name}')
        # load json and create model
        json_file = open(os.path.join(args.model_dir, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(args.model_dir, args.model_name))
        log('load trained model on MRI Dataset by Stephen Gregory')
    else:
        print(f'A file DOES exist with the name: {args.model_dir}/{args.model_name}')
        model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
        log('load trained model')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))

        # Create a List of Peak Signal-To-Noise ratios (PSNRs)
        psnrs = []

        # Create a List of Structural Similarities (SSIMs)
        ssims = []

        for image_name in os.listdir(os.path.join(args.set_dir, set_cur, 'CoregisteredBlurryImages')):
            if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):

                # 1. Load the Clear Image x (as grayscale), and standardize the pixel values, and..
                # 2. Save the original mean and standard deviation of x
                x, x_orig_mean, x_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         set_cur,
                                                                                         'ClearImages',
                                                                                         image_name), 0))

                # Load the Coregistered Blurry Image y (as grayscale), and standardize the pixel values, and...
                # 2. Save the original mean and standard deviation of y
                y, y_orig_mean, y_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         set_cur,
                                                                                         'CoregisteredBlurryImages',
                                                                                         image_name), 0))

                # Convert y from an image (numpy array) to a tensor
                y_tensor = to_tensor(y)

                # Start a timer
                start_time = time.time()

                # Inference (Denoise y_tensor to get x_pred)
                x_pred = model.predict(y_tensor)

                # Record the inference time
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, image_name, elapsed_time))

                # Converts x_pred from a tensor to an image (numpy array)
                x_pred = from_tensor(x_pred)

                ''' Just logging 
                # Reverse the standardization
                x_pred_reversed = image_utils.reverse_standardize(x_pred, original_mean=x_orig_mean, original_std=x_orig_std)
                x_reversed = image_utils.reverse_standardize(x, original_mean=x_orig_mean, original_std=x_orig_std)

                logger.show_images([("x", x),
                                    ("x_reversed", x_reversed),
                                    ("x_pred", x_pred),
                                    ("x_pred_reversed", x_pred_reversed),
                                    ("y", y)])
                '''

                # Reverse the standardization of x, x_pred, and y
                x = image_utils.reverse_standardize(x, original_mean=x_orig_mean, original_std=x_orig_std)
                x_pred = image_utils.reverse_standardize(x_pred, original_mean=x_orig_mean, original_std=x_orig_std)
                y = image_utils.reverse_standardize(y, original_mean=y_orig_mean, original_std=y_orig_std)

                # Get the PSNR and SSIM for x
                psnr_x = peak_signal_noise_ratio(x, x_pred)
                ssim_x = structural_similarity(x, x_pred, multichannel=True)

                # If we want to save the result...
                if args.save_result:
                    name, ext = os.path.splitext(image_name)

                    ''' Just logging
                    # Show the images
                    logger.show_images([("y", y),
                                        ("x_pred", x_pred)])
                    '''

                    # Then save the denoised image
                    cv2.imwrite(filename=os.path.join(args.result_dir, set_cur, name + '_dncnn' + ext), img=x_pred)

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
            # Save the result to <result_dir>/<set_cur>/results.txt
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))

        # Log the average PSNR and SSIM to the Terminal
        log('Datset: {0:10s} \n  Average PSNR = {1:2.2f}dB, Average SSIM = {2:1.4f}'.format(set_cur, psnr_avg,
                                                                                            ssim_avg))


if __name__ == '__main__':
    main()
