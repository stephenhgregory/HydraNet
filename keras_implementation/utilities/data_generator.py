import glob
import cv2
import numpy as np
from enum import Enum

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class ImageType(Enum):
    """An Enumerator representing a Clear Image and a Blurry Image"""
    CLEARIMAGE = 1
    BLURRYIMAGE = 2


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


def gen_patches(file_name):
    # read image
    image = cv2.imread(file_name, 0)  # gray scale
    h, w = image.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        image_scaled = cv2.resize(image, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = image_scaled[i:i + patch_size, j:j + patch_size]
                # patches.append(x)
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches


def data_generator(data_dir='data/train', image_type=ImageType.CLEARIMAGE, verbose=False):
    """
    Provides a numpy array of training examples, given a path to a training directory

    :param data_dir: The path of the training data directory
    :type data_dir: basestring
    :param image_type: The type of image that we wish to generate training data of
    :type image_type: ImageType
    :param verbose: Whether or not we want to log additional info about this file
    :type verbose: bool
    :return: training data
    :rtype: numpy.array
    """

    if image_type == ImageType.CLEARIMAGE:
        data_dir += '/CoregisteredImages'
    elif image_type == ImageType.BLURRYIMAGE:
        data_dir += '/BlurryImages'

    print(data_dir)

    file_list = glob.glob(data_dir + '/*.jpg')  # get name list of all .png files
    # initialize
    data = []
    # generate patches
    print(f'The length of the file list is: {len(file_list)}')
    for i in range(len(file_list)):
        patch = gen_patches(file_list[i])
        data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    discard_n = len(data) - len(data) // batch_size * batch_size;
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
