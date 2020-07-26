import glob
import cv2
import numpy as np
from enum import Enum
import os
from os.path import join
import keras_implementation.utilities.logger as logger
import keras_implementation.utilities.image_utils as image_utils

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class ImageType(Enum):
    """An Enumerator representing a Clear Image and a Blurry Image"""
    CLEARIMAGE = 1
    BLURRYIMAGE = 2


class ImageFormat(Enum):
    """An Enumerator representing whether an image is a JPG image or a PNG image"""
    JPG = 1
    PNG = 2


def show(x, title=None, cbar=False, figsize=None):
    """
    Shows an input image/set of images x"" using matplotlib

    :param x: The input image to be shown
    :param title: The title of the matplotlib plot
    :param cbar: (bool) Whether or not we want to render a colorbar
    :param figsize: The width/height in inches of the plot to be shown
    :return: None
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(image, mode=0):
    """
    A function providing multiple ways of augmenting an image
    
    :param image: An input image to be augmented
    :param mode: The specific augmentation to perform on the input image
    :return: The augmented image
    """
    if mode == 0:
        return image
    elif mode == 1:
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))


def generate_patches_from_file_name(file_name):
    """
    Generates and returns a list of image patches from an input file_name

    :param file_name: The name of the image to generate patches from
    :return: patches: A list of image patches
    """

    # Read the image as grayscale
    image = cv2.imread(file_name, 0)

    # Generate and return a list of patches from the image
    return generate_patches(image)


def generate_patches(image):
    """
    Generates and returns a list of image patches from an input image

    :param image: The image to generate patches from
    :return: patches: A list of image patches
    """

    # Get the height and width of the image
    height, width = image.shape

    # Store the patches in a list
    patches = []

    # For each scale
    for scale in scales:

        # Get the scaled height and width
        height_scaled, width_scaled = int(height * scale), int(width * scale)

        # Rescale the image
        image_scaled = cv2.resize(image, (height_scaled, width_scaled), interpolation=cv2.INTER_CUBIC)

        # Extract patches
        for i in range(0, height_scaled - patch_size + 1, stride):
            for j in range(0, width_scaled - patch_size + 1, stride):
                patch = image_scaled[i:i + patch_size, j:j + patch_size]
                patches.append(patch)

    return patches


def generate_augmented_patches(image):
    """
    Generates and returns a list of image patches from an input file_name,
    adding a random augmentation to each patch (flip, rotate, etc.)

    :param image: The image to generate patches from
    :return: patches: A list of image patches
    """

    # Get the height and width of the image
    height, width = image.shape

    # Store the patches in a list
    patches = []

    # For each scale
    for scale in scales:

        # Get the scaled height and width
        height_scaled, width_scaled = int(height * scale), int(width * scale)

        # Rescale the image
        image_scaled = cv2.resize(image, (height_scaled, width_scaled), interpolation=cv2.INTER_CUBIC)

        # Extract patches
        for i in range(0, height_scaled - patch_size + 1, stride):
            for j in range(0, width_scaled - patch_size + 1, stride):
                patch = image_scaled[i:i + patch_size, j:j + patch_size]

                # Augment the patch x before adding it to the list of patches
                for k in range(0, aug_times):
                    patch_aug = data_aug(patch, mode=np.random.randint(0, 8))
                    patches.append(patch_aug)

    return patches


def generate_augmented_patches_from_file_name(file_name):
    """
    Generates and returns a list of image patches from an input file_name,
    adding a random augmentation to each patch (flip, rotate, etc.)

    :param file_name: The name of the image to generate patches from
    :type file_name: str

    :return: patches: A list of image patches
    """

    # Read the image as grayscale
    image = cv2.imread(file_name, 0)

    # Generate a return a list of augmented patches from the image
    return generate_augmented_patches(image)


def data_generator(root_dir=join('data', 'Volume1', 'train'),
                   image_type=ImageType.CLEARIMAGE,
                   verbose=False,
                   image_format=ImageFormat.PNG):
    """
    Provides a numpy array of training examples, given a path to a training directory

    :param image_format: The format of image that our training data is (JPG or PNG)
    :type image_format: ImageFormat
    :param root_dir: The path of the training data directory
    :type root_dir: str
    :param image_type: The type of image that we wish to generate training data of
    :type image_type: ImageType
    :param verbose: Whether or not we want to log additional info about this file
    :type verbose: bool

    :return: training data
    :rtype: numpy.array
    """

    if image_type == ImageType.CLEARIMAGE:
        image_dir = join(root_dir, 'ClearImages')
    elif image_type == ImageType.BLURRYIMAGE:
        image_dir = join(root_dir, 'CoregisteredBlurryImages')

    # If data is PNGs, get the list of all .png files
    if image_format == ImageFormat.PNG:
        file_list = sorted(glob.glob(image_dir + '/*.png'))
    # Else if data is JPGs, get the list of all .jpg files
    elif image_format == ImageFormat.JPG:
        file_list = sorted(glob.glob(image_dir + '/*.jpg'))

    # initialize data list
    data = []

    # generate patches
    print(f'The length of the file list is: {len(file_list)}')

    # Iterate over the entire list of images
    for i, file_name in enumerate(os.listdir(image_dir)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):

            # Read the image as a numpy array
            image = cv2.imread(os.path.join(image_dir, file_name), 0)

            # If the image is a blurry image
            if image_type == ImageType.BLURRYIMAGE:
                # Read the corresponding Clear Image
                clear_image = cv2.imread(os.path.join(root_dir, 'ClearImages', file_name))

                # Histogram equalize the image
                equalized_image = image_utils.hist_match(image, image).astype('uint8')

                ''' Just logging
                # Show the blurry image (Pre-Histogram Equalization), clear image, and
                # blurry image (Post-Histogram Equalization)
                logger.show_images([(f'CoregisteredBlurryImage (Pre-Histogram Equalization)', image),
                                    (f'Matching Clear Image', clear_image),
                                    ('CoregisteredBlurryImage (Post-Histogram Equalization)', equalized_image)])
                '''

                # Set our blurry image equal to the histogram matched image
                image = equalized_image

            # Generate patches from the image
            patches = generate_patches(image)

            # Append the patches to data
            data.append(patches)
            if verbose:
                print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')

    # Convert data to a numpy array of ints
    data = np.array(data, dtype='uint8')

    # reshape data
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))

    # Get the number of elements of n to discard
    discard_n = len(data) - len(data) // batch_size * batch_size;

    # Remove the range of "discard_n" from data
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')

    return data


def data_generator_augmented(root_dir=join('data', 'Volume1', 'train'), image_type=ImageType.CLEARIMAGE, verbose=False, image_format=ImageFormat.PNG):
    """
    Provides a numpy array of training examples, given a path to a training directory.
    Adds augmentation (flipping, rotation, etc.) to the training examples for rotational and flipping invariance

    :param root_dir: The path of the training data directory
    :type root_dir: str
    :param image_type: The type of image that we wish to generate training data of
    :type image_type: ImageType
    :param verbose: Whether or not we want to log additional info about this file
    :type verbose: bool
    :return: training data
    :rtype: numpy.array
    """

    if image_type == ImageType.CLEARIMAGE:
        image_dir = join(root_dir, 'ClearImages')
    elif image_type == ImageType.BLURRYIMAGE:
        image_dir = join(root_dir, 'CoregisteredBlurryImages')

    # If data is PNGs, get the list of all .png files
    if image_format == ImageFormat.PNG:
        file_list = sorted(glob.glob(image_dir + '/*.png'))
    # Else if data is JPGs, get the list of all .jpg files
    elif image_format == ImageFormat.JPG:
        file_list = sorted(glob.glob(image_dir + '/*.jpg'))

    # initialize data list
    data = []

    print(data_dir)

    # Get the name list of all .jpg files
    file_list = glob.glob(data_dir + '/*.jpg')

    # initialize data list
    data = []

    # generate patches
    print(f'The length of the file list is: {len(file_list)}')
    for i in range(len(file_list)):
        patch = generate_augmented_patches_from_file_name(file_list[i])

        data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')

    # Convert data to a numpy array of ints
    data = np.array(data, dtype='uint8')

    # reshape data
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))

    # Get the number of elements of n to discard
    discard_n = len(data) - len(data) // batch_size * batch_size;

    # Remove the range of "discard_n" from data
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')

    return data


if __name__ == '__main__':
    data = data_generator(data_dir='data/Train400')

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')
