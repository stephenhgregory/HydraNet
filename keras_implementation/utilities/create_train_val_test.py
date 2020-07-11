import os
from os.path import isfile, join
import numpy as np
import shutil
import errno


def main(root_dir=(os.getcwd() + '/data')):
    create_train_test_val_dirs(root_dir)
    populate_train_test_val_dirs_nonrandomly(root_dir)


def create_train_test_val_dirs(root_dir=os.getcwd()):
    """
    Creates directories that will later hold train, validation, and test
    splits of image dataset.
    :param root_dir: The root directory under which the train, val, and test sets will live
    :return: None
    """
    try:
        os.makedirs(root_dir + '/train')
        os.makedirs(root_dir + '/train/CoregisteredImages')
        os.makedirs(root_dir + '/train/BlurryImages')
        os.makedirs(root_dir + '/val')
        os.makedirs(root_dir + '/val/CoregisteredImages')
        os.makedirs(root_dir + '/val/BlurryImages')
        os.makedirs(root_dir + '/test')
        os.makedirs(root_dir + '/test/CoregisteredImages')
        os.makedirs(root_dir + '/test/BlurryImages')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def print_file_distribution(num_imgs_total, num_imgs_train, num_imgs_val, num_imgs_test):
    """
    Prints the distribution of image files across all directories
    :param num_imgs_total: Total number of images in dataset
    :param num_imgs_train: Number of training images in dataset
    :param num_imgs_val: Number of validation images in dataset
    :param num_imgs_test: Number of test images in dataset
    :return: None
    """
    print('Total images: ', num_imgs_total)
    print('Training: ', num_imgs_train)
    print('Validation: ', num_imgs_val)
    print('Testing: ', num_imgs_test)


def populate_train_test_val_dirs_nonrandomly(root_dir=(os.getcwd()), val_ratio=0.15, test_ratio=0.05):
    """
    Populates the train, val, and test folders with the images located in root_dir,
    according to val_ratio  and test_ratio
    :param root_dir: The root directory of the image dataset
    :param val_ratio: The desired ratio of val images to total images
    :param test_ratio: The desired ratio of test images to total images
    :return: None
    """

    ''' Creating partitions of the data after shuffling '''
    # Folder to copy images from
    src = join(root_dir, 'CoregisteredImages')

    all_file_names = [f for f in os.listdir(src) if isfile(join(src, f))]

    # Select the number of images to skip between validation images
    val_skip_number = len(all_file_names) / (val_ratio * len(all_file_names))

    # Select the number of images to skip between test images
    test_skip_number = len(all_file_names) / (test_ratio * len(all_file_names))

    # Get the list of validation file names, test file names, and train file names
    val_file_names = all_file_names[::int(val_skip_number)]
    test_file_names = [filename for filename in all_file_names[::int(test_skip_number + 1)] if filename not in val_file_names]
    train_file_names = [filename for filename in all_file_names if filename not in val_file_names and filename not in test_file_names]

    ''' Print the file distribution amongst the folders '''
    print_file_distribution(len(all_file_names), len(train_file_names), len(val_file_names), len(test_file_names))

    print(train_file_names)

    ''' Copy-Pasting Images '''
    for name in train_file_names:
        shutil.copy(join(root_dir, 'CoregisteredImages', name), root_dir + '/train/CoregisteredImages')
        shutil.copy(join(root_dir, 'BlurryImages', name), root_dir + '/train/BlurryImages')
    for name in val_file_names:
        shutil.copy(join(root_dir, 'CoregisteredImages', name), root_dir + '/val/CoregisteredImages')
        shutil.copy(join(root_dir, 'BlurryImages', name), root_dir + '/val/BlurryImages')
    for name in test_file_names:
        shutil.copy(join(root_dir, 'CoregisteredImages', name), root_dir + '/test/CoregisteredImages')
        shutil.copy(join(root_dir, 'BlurryImages', name), root_dir + '/test/BlurryImages')


def populate_train_test_val_dirs_randomly(root_dir=(os.getcwd()), val_ratio=0.15, test_ratio=0.05):
    """
    Populates the train, val, and test folders with the images located in root_dir,
    according to val_ratio  and test_ratio
    :param root_dir: The root directory of the image dataset
    :param val_ratio: The desired ratio of val images to total images
    :param test_ratio: The desired ratio of test images to total images
    :return: None
    """

    ''' Creating partitions of the data after shuffling '''
    # Folder to copy images from
    src = root_dir  # The folder to copy images from

    all_file_names = [f for f in os.listdir(src) if isfile(join(src, f))]

    np.random.shuffle(all_file_names)

    train_file_names, val_file_names, test_file_names = np.split(np.array(all_file_names),
                                                                 [int(len(all_file_names) * (
                                                                         1 - val_ratio + test_ratio)),
                                                                  int(len(all_file_names) * (1 - test_ratio))])
    ''' Print the file distribution amongst the folders '''
    print_file_distribution(len(all_file_names), len(train_file_names), len(val_file_names), len(test_file_names))

    print(train_file_names)

    ''' Copy-Pasting Images '''
    for name in train_file_names:
        shutil.copy(join(root_dir, 'CoregisteredImages', name), root_dir + '/train/CoregisteredImages')
        shutil.copy(join(root_dir, 'BlurryImages', name), root_dir + '/train/BlurryImages')
    for name in val_file_names:
        shutil.copy(join(root_dir, 'CoregisteredImages', name), root_dir + '/val/CoregisteredImages')
        shutil.copy(join(root_dir, 'BlurryImages', name), root_dir + '/val/BlurryImages')
    for name in test_file_names:
        shutil.copy(join(root_dir, 'CoregisteredImages', name), root_dir + '/test/CoregisteredImages')
        shutil.copy(join(root_dir, 'BlurryImages', name), root_dir + '/test/BlurryImages')


if __name__ == "__main__":
    main()
