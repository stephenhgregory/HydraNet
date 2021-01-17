import glob
import cv2
import numpy as np
from enum import Enum
import os
from os.path import join
from typing import List, Tuple, Dict
from utilities import image_utils
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import zoom

# Global variable definitions
aug_times = 1
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


def generate_patches_from_file_name(file_name: str, patch_size: int = 40, stride: int = 10,
                                    scales: List[float] = [1, 0.9, 0.8, 0.7]):
    """
    Generates and returns a list of image patches from an input file_name

    :param file_name: The name of the image to generate patches from
    :param patch_size: The size of the image patches to produce -> (patch_size, patch_size)
    :type patch_size: int
    :param stride: The stride with which to slide the patch-taking window
    :type stride: int
    :param scales: A list of scales at which we want to create image patches.
        If None, this function simply performs no rescaling of the image to create patches
    :type scales: List

    :return: patches: A list of image patches
    """

    # Read the image as grayscale
    image = cv2.imread(file_name, 0)

    # Generate and return a list of patches from the image
    return generate_patch_pairs(image)


# def generate_patches_old(image, image_type):
#     """
#     Generates and returns a list of image patches from an input image
#
#     :param image: The image to generate patches from
#     :type image: numpy array
#     :param image_type: The type of the image, CLEARIMAGE or BLURRYIMAGE
#     :type image_type: ImageType
#
#     :return: patches: A list of image patches
#     """
#
#     # Get the height and width of the image
#     height, width = image.shape
#
#     # Store the patches in a list
#     patches = []
#
#     # For each scale
#     for scale in scales:
#
#         # Get the scaled height and width
#         height_scaled, width_scaled = int(height * scale), int(width * scale)
#
#         # Rescale the image
#         image_scaled = cv2.resize(image, (height_scaled, width_scaled), interpolation=cv2.INTER_CUBIC)
#
#         # Extract patches
#         for i in range(0, height_scaled - patch_size + 1, stride):
#             for j in range(0, width_scaled - patch_size + 1, stride):
#                 patch = image_scaled[i:i + patch_size, j:j + patch_size]
#                 patches.append(patch)
#
#     return patches


def generate_3d_patch_pairs(clear_volume: np.ndarray, blurry_volume: np.ndarray, patch_size: Tuple[int, int, int] = 40,
                            stride: int = 10, scales: List[float] = [1., 0.9, 0.8, 0.7]) -> Tuple[List, List]:
    """
    Generates lists of image patch volume pairs from a set of 3d image volumes
    (Not a generator)

    Parameters
    ----------
    clear_volume: The clear image volume to generate patches from
    blurry_volume: The blurry image volume to generate patches from
    patch_size: The size of the image patch volumes to produce -> (depth, height, width)
        TODO: Note that CV image dimension convention is (Channel, Height, Width) as opposed to (Height, Width, Channel)
        TODO: Make sure we're sticking with that!
    stride: The stride with which to sample the patch volume window
    scales: A list of scales at which we want to create image patch volumes.

    Returns
    -------
    A tuple of a list of image patch volumes, (clear_patches, blurry_patches). TODO: Maybe add more documentation here
    """

    # Make sure Clear Volume and Blurry Volume have the same shape
    assert (clear_volume.shape == blurry_volume.shape)

    # Get the volume, height, and width of the image TODO: Make sure we're sticking with (depth, height, width)
    depth, height, width = clear_volume.shape

    # Store the patch volumes in a list
    clear_patches = []
    blurry_patches = []

    # For each scale
    for scale in scales:
        # Get the scaled depth, height and width TODO: Make sure we're sticking with (depth, height, width)
        depth_scaled, height_scaled, width_scaled = int(depth * scale), int(height * scale), int(width * scale)

        # Rescale the images TODO: Make sure we're sticking with (depth, height, width)
        clear_volume_scaled = zoom(clear_volume, (scale, scale, scale))
        blurry_volume_scaled = zoom(blurry_volume, (scale, scale, scale))

        # Extract patches TODO: Make sure we're sticking with (depth, height, width)
        for i in range(0, depth_scaled - patch_size[0] + 1, stride):
            for j in range(0, height_scaled - patch_size[1] + 1, stride):
                for k in range(0, width_scaled - patch_size[2] + 1, stride):
                    # Get the clear and blurry patches
                    clear_patch = clear_volume_scaled[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                    blurry_patch = blurry_volume_scaled[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]

                    # Add the clear_patch and blurry_patch to clear_patches and blurry_patches, respectively
                    clear_patches.append(clear_patch)
                    blurry_patches.append(blurry_patch)

    return clear_patches, blurry_patches


def generate_patch_pairs(clear_image: np.ndarray, blurry_image: np.ndarray, patch_size: int = 40, stride: int = 10,
                         scales: List[float] = [1, 0.9, 0.8, 0.7]):
    """
    Generates and returns a list of image patches from an input image

    :param clear_image: The clear image to generate patches from
    :type clear_image: numpy array
    :param blurry_image: The blurry image to generate patches from
    :type blurry_image: numpy array
    :param patch_size: The size of the image patches to produce -> (patch_size, patch_size)
    :type patch_size: int
    :param stride: The stride with which to slide the patch-taking window
    :type stride: int
    :param scales: A list of scales at which we want to create image patches.
        If None, this function simply performs no rescaling of the image to create patches
    :type scales: List

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

    if scales is None:
        # Extract patches
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                # Get the clear and blurry patches
                clear_patch = clear_image[i:i + patch_size, j:j + patch_size]
                blurry_patch = blurry_image[i:i + patch_size, j:j + patch_size]

                # Add the clear_patch and blurry_patch to clear_patches and blurry_patches, respectively
                clear_patches.append(clear_patch)
                blurry_patches.append(blurry_patch)

        return clear_patches, blurry_patches

    else:
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


def generate_augmented_patches(image: np.ndarray, patch_size: int = 40, stride: int = 10,
                               scales: List[float] = [1, 0.9, 0.8, 0.7]):
    """
    Generates and returns a list of image patches from an input file_name,
    adding a random augmentation to each patch (flip, rotate, etc.)

    :param image: The image to generate patches from
    :param patch_size: The size of each patches in pixels -> (patch_size, patch_size)
    :type patch_size: int
    :param stride: The stride with which to slide the patch-taking window
    :type stride: int
    :param scales: A list of scales at which we want to create image patches.
        If None, this function simply performs no rescaling of the image to create patches
    :type scales: List

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


def gen_patches(file_name, scales=[1, 0.9, 0.8, 0.7], patch_size=40, stride=10):
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


def get_lower_and_upper_percentile_stds(data_dir: str, lower_percentile: float, upper_percentile: float,
                                        patch_size: int = 40, stride: int = 20, scales: List[float] = None):
    """
    Gets the lower and upper percentile values of std of the images in a given data directory

    :param data_dir: The directory housing input images
    :param lower_percentile: The percentile at which to find the corresponding lower percentile PSNR value
    :param upper_percentile: The percentile at which to find the corresponding hight percentile PSNR value
    :param patch_size: The patch size (in pixels) of each image patch taken
    :param stride: The stride with which to slide the patch-taking window
    :param scales: A list of scales at which we want to create image patches.
        If None, this function simply performs no rescaling of the image to create patches

    :return: (lower_percentile_value, upper_percentile_value)
    """
    # Get training examples from data_dir using data_generator
    _, y_original = pair_data_generator(data_dir, patch_size=patch_size, stride=stride, scales=scales)

    # Iterate over y_original and get stds
    stds = []
    for y_patch in y_original:
        if np.max(y_patch) < 10:
            continue
        y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1])
        std = np.std(y_patch)
        stds.append(std)

    stds = np.array(stds, dtype='float64')

    return np.percentile(stds, lower_percentile), np.percentile(stds, upper_percentile)


def pair_3d_data_generator(root_dirs: str = join('data', 'Volume1', 'train'),
                           patch_size: Tuple[int, int, int] = (20, 20, 20), stride: int = 10,
                           scales: List[float] = [1, 0.9, 0.8, 0.7]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Provides a numpy array of training examples, given paths to (one or many) training directory

    Parameters
    ----------
    root_dirs: A list of paths to training data directories
    patch_size: The size of each patches in pixels -> (patch_size, patch_size)
    stride: The stride with which to slide the patch-taking window
    scales: A list of scales at which we want to create image patches.

    Returns
    -------
    (clear_volume_patches, blurry_volume_patches)
    """
    all_clear_volume_patches = []
    all_blurry_volume_patches = []

    for root_dir in root_dirs:
        # Obtain 3D image volumes
        clear_image_volume = image_utils.get_3d_image_volume(image_dir=join(root_dir, 'ClearImages'))
        blurry_image_volume = image_utils.get_3d_image_volume(image_dir=join(root_dir, 'CoregisteredBlurryImages'))

        # Histogram equalize the blurry volume px distribution to match the clear image px distribution
        blurry_image_volume = image_utils.hist_match(source=blurry_image_volume, template=clear_image_volume).astype(
            'uint8')

        # Generate clear and blurry patches from the clear and blurry images, respectively...
        clear_volume_patches, blurry_volume_patches = generate_3d_patch_pairs(clear_volume=clear_image_volume,
                                                                              blurry_volume=blurry_image_volume,
                                                                              patch_size=patch_size, stride=stride,
                                                                              scales=scales)

        # Append the patches to clear_data and blurry_data
        all_clear_volume_patches.extend(clear_volume_patches)
        all_blurry_volume_patches.extend(blurry_volume_patches)

    # Convert clear_patches and blurry_patches to numpy arrays of ints
    all_clear_volume_patches = np.array(all_clear_volume_patches, dtype='uint8')
    all_blurry_volume_patches = np.array(all_blurry_volume_patches, dtype='uint8')

    # Extend the dimensionality of the patches by adding a new dimension
    all_clear_volume_patches = all_clear_volume_patches[..., np.newaxis]
    all_blurry_volume_patches = all_blurry_volume_patches[..., np.newaxis]

    # Make sure that clear_data and blurry_data have the same shape
    assert (all_clear_volume_patches.shape == all_blurry_volume_patches.shape)

    return all_clear_volume_patches, all_blurry_volume_patches


def cleanup_data_generator(root_dirs: str = join('data', 'subj1', 'train'),
                           image_format = ImageFormat.PNG) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    root_dirs: The directories under which data is found
    image_format: The type of image format used (PNG, JPG, etc.)

    Return
    ------
    (clear_data, blurry_data)
    """
    clear_data = []
    blurry_data = []

    for root_dir in root_dirs:

        # Get the directory name for the Clear and Blurry Images
        clear_image_dir = join(root_dir, 'ClearImages')
        blurry_image_dir = join(root_dir, 'CoregisteredBlurryImages')

        # Iterate over the entire list of images
        for i, file_name in enumerate(os.listdir(clear_image_dir)):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                # Read the Clear and Blurry Images as numpy arrays
                clear_image = cv2.imread(os.path.join(clear_image_dir, file_name), 0)
                blurry_image = cv2.imread(os.path.join(blurry_image_dir, file_name), 0)

                # Histogram equalize the blurry image px distribution to match the clear image px distribution
                blurry_image = image_utils.hist_match(blurry_image, clear_image).astype('uint8')

                # Append the images to full lists of data
                clear_data.extend(clear_image)
                blurry_data.extend(blurry_image)

                ''' Just logging 
                # Show the blurry image (Pre-Histogram Equalization), clear image, and
                # blurry image (Post-Histogram Equalization)
                logger.show_images([(f'CoregisteredBlurryImage (Pre-Histogram Equalization)', image),
                                    (f'Matching Clear Image', clear_image),
                                    ('CoregisteredBlurryImage (Post-Histogram Equalization)', equalized_image)])
                '''

    # Convert clear_data and blurry_data to numpy arrays of ints
    clear_data = np.array(clear_data, dtype='uint8')
    blurry_data = np.array(blurry_data, dtype='uint8')

    # Extend the dimensionality of the images by adding a new dimension
    clear_data = clear_data[..., np.newaxis]
    blurry_data = blurry_data[..., np.newaxis]

    # Make sure that clear_data and blurry_data have the same length
    assert (len(blurry_data) == len(clear_data))

    return clear_data, blurry_data


def pair_data_generator(root_dir: str = join('data', 'Volume1', 'train'), image_format: ImageFormat = ImageFormat.PNG,
                        patch_size: int = 40, stride: int = 10,
                        scales: List[float] = [1, 0.9, 0.8, 0.7]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Provides a numpy array of training examples, given a path to a training directory

    :param image_format: The format of image that our training data is (JPG or PNG)
    :param root_dir: The path of the training data directory
    :param patch_size: The size of each patches in pixels -> (patch_size, patch_size)
    :param stride: The stride with which to slide the patch-taking window
    :param scales: A list of scales at which we want to create image patches.
        If None, this function simply performs no rescaling of the image to create patches

    :return: (clear_data, blurry_data)
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
                                                                 blurry_image=blurry_image,
                                                                 patch_size=patch_size,
                                                                 stride=stride,
                                                                 scales=scales)

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


def retrieve_train_data(train_data_dir: str, low_noise_threshold: float = 0.03, high_noise_threshold: float = 0.15,
                        skip_every: int = 3, patch_size: int = 40, stride: int = 10, scales: List = [1],
                        similarity_metric: str = 'psnr') -> Dict:
    """
    Gets and returns the image patches used during training time, split into 3 noise levels.
    Used to cross-reference patches at inference time.

    NOTE: Also returns PSNR of each patch.

    Parameters
    ----------
    train_data_dir: The root directory of the training data
    low_noise_threshold: The lower residual image standard deviation threshold used to determine which data
                                should go to which network
    high_noise_threshold: The upper residual image standard deviation threshold used to determine which data
                                should go to which network
    skip_every: If 1, we skip every 'skip_every' number of patches, and so return a smaller subset of patches
    patch_size: The size of each patches in pixels -> (patch_size, patch_size)
    stride: The stride with which to slide the patch-taking window
    scales: A list of scales at which we want to create image patches. If None, this function simply performs no
                                rescaling of the image to create patches

    Returns
    -------
    A dictionary of the following:
                1. x_low_noise: the clear patches at a low noise level
                2. y_low_noise: the blurry patches at a low noise level
                3. comparison_metrics_low_noise: the standard deviation of the residuals at a low noise level
                4. x_medium_noise: the clear patches at a medium noise level
                5. y_medium_noise: the blurry patches at a medium noise level
                6. comparison_metrics_medium_noise: the standard deviation of the residuals at a medium noise level
                7. x_high_noise: the clear patches at a high noise level
                8. y_high_noise: the blurry patches at a high noise level
                9. comparison_metrics_high_noise: the standard deviation of the residuals at a high noise level
        """

    print(f'Accessing training data in: {train_data_dir}')

    # Get training examples from data_dir using data_generator
    x, y = pair_data_generator(train_data_dir, patch_size=patch_size, stride=stride, scales=scales)

    # Create lists to store all of the patches and comparison metrics for each noise level category
    x_low_noise = []
    y_low_noise = []
    comparison_metrics_low_noise = []
    x_medium_noise = []
    y_medium_noise = []
    comparison_metrics_medium_noise = []
    x_high_noise = []
    y_high_noise = []
    comparison_metrics_high_noise = []

    # Iterate over all of the image patches
    for x_patch, y_patch in zip(x, y):

        # If the patch is black (i.e. the max px value < 10), just skip this training example
        if np.max(x_patch) < 10:
            continue

        if similarity_metric == 'std':
            # Get the residual std
            comparison_metric = get_residual_std(clear_patch=x_patch, blurry_patch=y_patch)
            # Add the patches and their residual stds to their corresponding lists based on noise level
            if comparison_metric < low_noise_threshold:
                x_low_noise.append(x_patch)
                y_low_noise.append(y_patch)
                comparison_metrics_low_noise.append(comparison_metric)
                continue
            elif low_noise_threshold < comparison_metric < high_noise_threshold:
                x_medium_noise.append(x_patch)
                y_medium_noise.append(y_patch)
                comparison_metrics_medium_noise.append(comparison_metric)
                continue
            elif comparison_metric > high_noise_threshold:
                x_high_noise.append(x_patch)
                y_high_noise.append(y_patch)
                comparison_metrics_high_noise.append(comparison_metric)
                continue

        elif similarity_metric == 'psnr':
            # Get the PSNR
            comparison_metric = peak_signal_noise_ratio(image_true=x_patch, image_test=y_patch)
            # Add the patches and their PSNRs to their corresponding lists based on noise level
            if comparison_metric < low_noise_threshold:
                x_high_noise.append(x_patch)
                y_high_noise.append(y_patch)
                comparison_metrics_high_noise.append(comparison_metric)
                continue
            elif low_noise_threshold < comparison_metric < high_noise_threshold:
                x_medium_noise.append(x_patch)
                y_medium_noise.append(y_patch)
                comparison_metrics_medium_noise.append(comparison_metric)
                continue
            elif comparison_metric > high_noise_threshold:
                x_low_noise.append(x_patch)
                y_low_noise.append(y_patch)
                comparison_metrics_low_noise.append(comparison_metric)
                continue

        elif similarity_metric == 'ssim':
            # Get the PSNR
            comparison_metric, _ = structural_similarity(x_patch, y_patch, full=True)
            # Add the patches and their PSNRs to their corresponding lists based on noise level
            if comparison_metric < low_noise_threshold:
                x_high_noise.append(x_patch)
                y_high_noise.append(y_patch)
                comparison_metrics_high_noise.append(comparison_metric)
                continue
            elif low_noise_threshold < comparison_metric < high_noise_threshold:
                x_medium_noise.append(x_patch)
                y_medium_noise.append(y_patch)
                comparison_metrics_medium_noise.append(comparison_metric)
                continue
            elif comparison_metric > high_noise_threshold:
                x_low_noise.append(x_patch)
                y_low_noise.append(y_patch)
                comparison_metrics_low_noise.append(comparison_metric)
                continue

    # Convert image patches and stds into numpy arrays
    x_low_noise = np.array(x_low_noise[::skip_every], dtype='uint8')
    y_low_noise = np.array(y_low_noise[::skip_every], dtype='uint8')
    comparison_metrics_low_noise = np.array(comparison_metrics_low_noise[::skip_every], dtype='float64')
    x_medium_noise = np.array(x_medium_noise[::skip_every], dtype='uint8')
    y_medium_noise = np.array(y_medium_noise[::skip_every], dtype='uint8')
    comparison_metrics_medium_noise = np.array(comparison_metrics_medium_noise[::skip_every], dtype='float64')
    x_high_noise = np.array(x_high_noise[::skip_every], dtype='uint8')
    y_high_noise = np.array(y_high_noise[::skip_every], dtype='uint8')
    comparison_metrics_high_noise = np.array(comparison_metrics_high_noise[::skip_every], dtype='float64')

    training_patches = {
        "low_noise": {
            "x": x_low_noise,
            "y": y_low_noise,
            "comparison_metrics": comparison_metrics_low_noise
        },
        "medium_noise": {
            "x": x_medium_noise,
            "y": y_medium_noise,
            "comparison_metrics": comparison_metrics_medium_noise
        },
        "high_noise": {
            "x": x_high_noise,
            "y": y_high_noise,
            "comparison_metrics": comparison_metrics_high_noise
        }
    }

    # Return all of the patches and stds for the 3 categories
    return training_patches


if __name__ == '__main__':
    # data = pair_data_generator_multiple_data_dirs(root_dirs=['../data/subj1/train', '../data/subj2/train'])
    # data = pair_data_generator(root_dir='/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/data/subj1/train')
    # data = pair_3d_data_generator(root_dir='/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/data/subj1/train')
    data = pair_3d_data_generator(
        root_dirs=['/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/data/subj1/train',
                   '/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/data/subj2/train'])
