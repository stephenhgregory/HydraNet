import glob
import cv2
import numpy as np
from enum import Enum
import os
from os.path import join
from typing import List, Tuple, Dict
from utilities import image_utils

# Global variable definitions
patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class ImageType(Enum):
    """An Enum representing a Clear Image and a Blurry Image"""
    CLEARIMAGE = 1
    BLURRYIMAGE = 2


class NoiseLevel(Enum):
    """An Enum representing Low, Medium, or High Noise level"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ALL = 4


class ImageFormat(Enum):
    """An Enum representing whether an image is a JPG image or a PNG image"""
    JPG = 1
    PNG = 2


class ImagePatch:
    """
    Represents an image patch, which contains both an image patch as a numpy array, and
    a standard deviation of its residual as a float
    """

    def __init__(self, patch, std, image_type):
        """
        Constructor for ImagePatch

        :param patch: The image patch, a numpy array of pixel values
        :type patch: numpy array
        :param std: The standard deviation of the residual between the clear patch and corresponding blurry patch
        :type std: float
        :param image_type: The type of this ImagePatch, whether it be a Clear patch or a Blurry patch
        :type image_type: ImageType
        """
        self.image_type = image_type
        self.patch = patch
        self.std = std

    def convert_patch_to_int(self):
        """
        Converts self.patch to numpy array of ints

        :return: None
        """
        self.patch = np.array(self.patch, dtype='uint8')

    def reshape_patch(self):
        """
        Reshapes self.patch

        :return: None
        """
        self.patch = self.patch.reshape((self.patch.shape[0] * self.patch.shape[1],
                                         self.patch.shape[2],
                                         self.patch.shape[3],
                                         1))


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
    return generate_patch_pairs(image)


def generate_patches_old(image, image_type):
    """
    Generates and returns a list of image patches from an input image

    :param image: The image to generate patches from
    :type image: numpy array
    :param image_type: The type of the image, CLEARIMAGE or BLURRYIMAGE
    :type image_type: ImageType

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


def generate_patch_pairs(clear_image, blurry_image):
    """
    Generates and returns a list of image patches from an input image

    :param clear_image: The clear image to generate patches from
    :type clear_image: numpy array
    :param blurry_image: The blurry image to generate patches from
    :type blurry_image: numpy array

    :return: (clear_patches, blurry_patches): A tuple of a list of ImagePatches.
                Each ImagePatch in the list of ImagePatches contains an image patch and the standard deviation
                of the pixels in that image patch.
    :rtype: tuple
    """

    # Make sure clear_image and blurry_image share the same shape
    assert (clear_image.shape == blurry_image.shape)

    # Get the height and width of the image
    height, width = clear_image.shape

    # Store the patches in a list
    clear_patches = []
    blurry_patches = []

    # For each scale
    for scale in scales:

        # Get the scaled height and width
        height_scaled, width_scaled = int(height * scale), int(width * scale)

        # Rescale the images
        clear_image_scaled = cv2.resize(clear_image, (height_scaled, width_scaled), interpolation=cv2.INTER_CUBIC)
        blurry_image_scaled = cv2.resize(blurry_image, (height_scaled, width_scaled), interpolation=cv2.INTER_CUBIC)

        # Extract patches
        for i in range(0, height_scaled - patch_size + 1, stride):
            for j in range(0, width_scaled - patch_size + 1, stride):
                # Get the clear and blurry patches
                clear_patch = clear_image_scaled[i:i + patch_size, j:j + patch_size]
                blurry_patch = blurry_image_scaled[i:i + patch_size, j:j + patch_size]

                # Add the clear_patch and blurry_patch to clear_patches and blurry_patches, respectively
                clear_patches.append(clear_patch)
                blurry_patches.append(blurry_patch)

    return clear_patches, blurry_patches


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


def get_residual_std(clear_patch, blurry_patch):
    """
    Gets the standard deviation of the residual between two image patches

    :param clear_patch: The Clear Image patch
    :type clear_patch: numpy array
    :param blurry_patch: The Blurry Image patch
    :type blurry_patch: numpy array

    :return:
    """
    residual = image_utils.get_residual(clear_image=clear_patch, blurry_image=blurry_patch)
    return np.std(residual)


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


def separate_images_and_stds(patches_and_stds):
    """
    Separates a numpy array of image patches and residual standard deviations
    into separate numpy arrays of patches and standard deviations, respectively

    :param patches_and_stds: 2D Numpy array of image patches and stds, where patches are the first
                                column of the array, and stds are the second column of the array

    :return: (patches, stds): a numpy array of image patches and a numpy array of residual stds
    :rtype: tuple
    """
    # Get patches and stds
    patches = patches_and_stds[:, 0]
    stds = patches_and_stds[:, 1]
    return patches, stds


def gen_patches(file_name):
    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                # patches.append(x)
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches


def datagenerator(data_dir=join('data', 'Volume1', 'train'), image_type=ImageType.CLEARIMAGE):
    """
    Provides a numpy array of training examples, given a path to a training directory

    :param data_dir: The directory in which the images are located
    :type data_dir: str
    :param image_type: The type of image that we wish to find, can be CLEARIMAGE or COREGISTEREDBLURRYIMAGE
    :type image_type: ImageType

    :return:
    """
    if image_type == ImageType.CLEARIMAGE:
        data_dir += '/ClearImages'
    elif image_type == ImageType.BLURRYIMAGE:
        data_dir += '/CoregisteredBlurryImages'

    print(data_dir)

    file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files

    # initialize
    data = []

    # generate patches
    print(f'The length of the file list is: {len(file_list)}')
    for i in range(len(file_list)):
        patch = gen_patches(file_list[i])
        data.append(patch)
    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))

    ''' Commenting this out, as it is done in train_datagen
    # Remove elements from data so that it has the right number of patches 
    discard_n = len(data) - len(data) // batch_size * batch_size;
    data = np.delete(data, range(discard_n), axis=0)
    '''

    print('^_^-training data finished-^_^')
    return data


def pair_data_generator(root_dir=join('data', 'Volume1', 'train'),
                        image_format=ImageFormat.PNG):
    """
    Provides a numpy array of training examples, given a path to a training directory

    :param image_format: The format of image that our training data is (JPG or PNG)
    :type image_format: ImageFormat
    :param root_dir: The path of the training data directory
    :type root_dir: str

    :return: training data
    :rtype: numpy.array
    """

    # Get the directory name for the Clear and Blurry Images
    clear_image_dir = join(root_dir, 'ClearImages')
    blurry_image_dir = join(root_dir, 'CoregisteredBlurryImages')

    # If data is PNGs, get the list of all .png files
    if image_format == ImageFormat.PNG:
        file_list = sorted(glob.glob(clear_image_dir + '/*.png'))
    # Else if data is JPGs, get the list of all .jpg files
    elif image_format == ImageFormat.JPG:
        file_list = sorted(glob.glob(clear_image_dir + '/*.jpg'))

    # initialize clear_data and clurry_data lists
    clear_data = []
    blurry_data = []

    # Iterate over the entire list of images
    for i, file_name in enumerate(os.listdir(clear_image_dir)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Read the Clear and Blurry Images as numpy arrays
            clear_image = cv2.imread(os.path.join(clear_image_dir, file_name), 0)
            blurry_image = cv2.imread(os.path.join(blurry_image_dir, file_name), 0)

            # Histogram equalize the blurry image px distribution to match the clear image px distribution
            blurry_image = image_utils.hist_match(blurry_image, clear_image).astype('uint8')

            ''' Just logging 
            # Show the blurry image (Pre-Histogram Equalization), clear image, and
            # blurry image (Post-Histogram Equalization)
            logger.show_images([(f'CoregisteredBlurryImage (Pre-Histogram Equalization)', image),
                                (f'Matching Clear Image', clear_image),
                                ('CoregisteredBlurryImage (Post-Histogram Equalization)', equalized_image)])
            '''

            # Generate clear and blurry patches from the clear and blurry images, respectively...
            clear_patches, blurry_patches = generate_patch_pairs(clear_image=clear_image,
                                                                 blurry_image=blurry_image)

            # Append the patches to clear_data and blurry_data
            clear_data.append(clear_patches)
            blurry_data.append(blurry_patches)

    # Convert clear_data and blurry_data to numpy arrays of ints
    clear_data = np.array(clear_data, dtype='uint8')
    blurry_data = np.array(blurry_data, dtype='uint8')

    # Reshape clear_data, blurry_data, and std_data
    clear_data = clear_data.reshape((clear_data.shape[0] * clear_data.shape[1],
                                     clear_data.shape[2],
                                     clear_data.shape[3],
                                     1
                                     ))
    blurry_data = blurry_data.reshape((blurry_data.shape[0] * blurry_data.shape[1],
                                       blurry_data.shape[2],
                                       blurry_data.shape[3],
                                       1
                                       ))

    # Make sure that clear_data and blurry_data have the same length
    assert (len(blurry_data) == len(clear_data))

    ''' Commenting this out, as it will be done in train_datagen
    # Get the number of elements to discard
    num_elements_to_discard = len(clear_data) - len(clear_data) // batch_size * batch_size
    # Remove "num_elements_to_discard" elements from from clear_data and blurry_data
    clear_data = np.delete(clear_data, range(num_elements_to_discard), axis=0)
    blurry_data = np.delete(blurry_data, range(num_elements_to_discard), axis=0)
    '''

    return clear_data, blurry_data


def retrieve_train_data(train_data_dir: str, low_noise_threshold: float = 0.04, high_noise_threshold: float = 0.15,
                        skip_every: int = 3) -> Dict:
    """
    Gets and returns the image patches used during training time, split into 3 noise levels.
    Used to cross-reference patches at inference time.

    :param train_data_dir: The root directory of the training data
    :type train_data_dir: str
    :param low_noise_threshold: The lower residual image standard deviation threshold used to determine which data
                                should go to which network
    :type low_noise_threshold: float
    :param high_noise_threshold: The upper residual image standard deviation threshold used to determine which data
                                should go to which network
    :type high_noise_threshold: float
    :param skip_every: If 1, we skip every 'skip_every' number of patches, and so return a smaller subset of patches

    :return: A dictionary of the following:
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
    x, y = pair_data_generator(train_data_dir)

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
        std = get_residual_std(clear_patch=x_patch,
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
    x_low_noise = np.array(x_low_noise[::skip_every], dtype='uint8')
    y_low_noise = np.array(y_low_noise[::skip_every], dtype='uint8')
    stds_low_noise = np.array(stds_low_noise[::skip_every], dtype='float64')
    x_medium_noise = np.array(x_medium_noise[::skip_every], dtype='uint8')
    y_medium_noise = np.array(y_medium_noise[::skip_every], dtype='uint8')
    stds_medium_noise = np.array(stds_medium_noise[::skip_every], dtype='float64')
    x_high_noise = np.array(x_high_noise[::skip_every], dtype='uint8')
    y_high_noise = np.array(y_high_noise[::skip_every], dtype='uint8')
    stds_high_noise = np.array(stds_high_noise[::skip_every], dtype='float64')

    training_patches = {
        "low_noise": {
            "x": x_low_noise,
            "y": y_low_noise,
            "stds": stds_low_noise
        },
        "medium_noise": {
            "x": x_medium_noise,
            "y": y_medium_noise,
            "stds": stds_medium_noise
        },
        "high_noise": {
            "x": x_high_noise,
            "y": y_high_noise,
            "stds": stds_high_noise
        }
    }

    # Return all of the patches and stds for the 3 categories
    return training_patches


def pair_data_generator_multiple_data_dirs(root_dirs,
                                           image_format=ImageFormat.PNG):
    """
    Provides a numpy array of training examples, given pahths to multiple training directories

    :param root_dirs: The paths of the training data directories
    :type root_dirs: list
    :param image_format: The format of image that our training data is (JPG or PNG)
    :type image_format: ImageFormat

    :return: training data
    :rtype: numpy.array
    """

    # initialize clear_data and clurry_data lists
    clear_data = []
    blurry_data = []

    # Iterate over the entire list of images
    for root_dir in root_dirs:
        print(f'Accessing training data in {root_dir}')
        for i, file_name in enumerate(os.listdir(join(root_dir, 'ClearImages'))):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                # Read the Clear and Blurry Images as numpy arrays
                clear_image = cv2.imread(os.path.join(root_dir, 'ClearImages', file_name), 0)
                blurry_image = cv2.imread(os.path.join(root_dir, 'CoregisteredBlurryImages', file_name), 0)

                # Histogram equalize the blurry image px distribution to match the clear image px distribution
                blurry_image = image_utils.hist_match(blurry_image, clear_image).astype('uint8')

                ''' Just logging 
                # Show the blurry image (Pre-Histogram Equalization), clear image, and
                # blurry image (Post-Histogram Equalization)
                logger.show_images([(f'CoregisteredBlurryImage (Pre-Histogram Equalization)', image),
                                    (f'Matching Clear Image', clear_image),
                                    ('CoregisteredBlurryImage (Post-Histogram Equalization)', equalized_image)])
                '''

                # Generate clear and blurry patches from the clear and blurry images, respectively...
                clear_patches, blurry_patches = generate_patch_pairs(clear_image=clear_image,
                                                                     blurry_image=blurry_image)

                # Append the patches to clear_data and blurry_data
                clear_data.append(clear_patches)
                blurry_data.append(blurry_patches)

    # Convert clear_data and blurry_data to numpy arrays of ints
    clear_data = np.array(clear_data, dtype='uint8')
    blurry_data = np.array(blurry_data, dtype='uint8')

    # Reshape clear_data, blurry_data, and std_data
    clear_data = clear_data.reshape((clear_data.shape[0] * clear_data.shape[1],
                                     clear_data.shape[2],
                                     clear_data.shape[3],
                                     1
                                     ))
    blurry_data = blurry_data.reshape((blurry_data.shape[0] * blurry_data.shape[1],
                                       blurry_data.shape[2],
                                       blurry_data.shape[3],
                                       1
                                       ))

    # Make sure that clear_data and blurry_data have the same length
    assert (len(blurry_data) == len(clear_data))

    ''' Commenting this out, as it will be done in train_datagen
    # Get the number of elements to discard
    num_elements_to_discard = len(clear_data) - len(clear_data) // batch_size * batch_size
    # Remove "num_elements_to_discard" elements from from clear_data and blurry_data
    clear_data = np.delete(clear_data, range(num_elements_to_discard), axis=0)
    blurry_data = np.delete(blurry_data, range(num_elements_to_discard), axis=0)
    '''

    return clear_data, blurry_data


if __name__ == '__main__':
    data = pair_data_generator_multiple_data_dirs(root_dirs=['../data/subj1/train', '../data/subj2/train'])
