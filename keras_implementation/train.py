"""
This file is used to train MyDenoiser
"""

import argparse
import re
import os
import glob
import numpy as np
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam
from keras_implementation.utilities import data_generator, logger, model_functions, image_utils
from keras_implementation.utilities.data_generator import NoiseLevel
import keras.backend as K

import tensorflow as tf

# Allow memory growth in order to fix a Tensorflow bug
physical_devices = tf.config.list_physical_devices('GPU')

# This makes sure that at runtime, the initialization of the CUDA device physical_devices[0] (The only GPU in
# the system) will not allocate ALL of the memory on that device.
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MyDnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Volume1/train', type=str, help='path of train data')
parser.add_argument('--val_data', default='data/Volume1/val', type=str, help='path of val data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=300, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1000, type=int, help='save model at after seeing x batches')
args = parser.parse_args()

save_dir = os.path.join('/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation',
                        'models',
                        args.model)

# Create the <save_dir> folder if it doesn't exist already
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def findLastCheckpoint(save_dir):
    """
    Finds the most recent Model checkpoint files

    :param save_dir:
    :return:
    """
    file_list = glob.glob(os.path.join(save_dir, 'model_*.hdf5'))  # get name list of all .hdf5 files
    # file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*", file_)
            # print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def lr_schedule(epoch):
    """
    Learning rate scheduler for tensorflow API

    :param epoch: The current epoch
    :type epoch: int
    :return: The Learning Rate
    :rtype: float
    """

    initial_lr = args.lr
    if epoch <= 30:
        lr = initial_lr
    elif epoch <= 60:
        lr = initial_lr / 10
    elif epoch <= 80:
        lr = initial_lr / 20
    else:
        lr = initial_lr / 20
    logger.log('current learning rate is %2.8f' % lr)
    return lr


def my_train_datagen(epoch_iter=2000, num_epochs=5, batch_size=128, data_dir=args.train_data,
                     noise_level=NoiseLevel.LOW,
                     low_noise_threshold=0.05,
                     high_noise_threshold=0.15):
    """
    Generator function that yields training data samples from a specified data directory

    :param epoch_iter: The number of iterations per epoch
    :param num_epochs: The total number of epochs
    :param batch_size: The number of training examples for each training iteration
    :param data_dir: The directory in which training examples are stored
    :param noise_level: The level of noise of the training data that we want
    :type noise_level: NoiseLevel
    :param low_noise_threshold: The lower residual image standard deviation threshold used to determine which data
                                should go to which network
    :type low_noise_threshold: float
    :param high_noise_threshold: The upper residual image standard deviation threshold used to determine which data
                                should go to which network
    :type high_noise_threshold: float

    :return: Yields a training example x and noisy image y
    """
    # Loop the following indefinitely...
    while True:
        # Set a counter variable
        counter = 0

        # If this is the first iteration...
        if counter == 0:
            print(f'Accessing training data in: {data_dir}')

            # Get training examples from data_dir using data_generator
            x_original, y_original = data_generator.pair_data_generator(data_dir, noise_level)

            # Create a list to hold all of the residual stds, and a list to hold the filtered x_original and y_original
            stds = []
            x_filtered = []
            y_filtered = []

            # Iterate over all of the image patches
            for x_patch, y_patch in zip(x_original, y_original):

                # If the patch is black (i.e. the max px value < 10), just skip this training example
                if np.max(x_patch) < 10:
                    continue

                # Get the residual std
                std = data_generator.get_residual_std(clear_patch=x_patch,
                                                      blurry_patch=y_patch)

                # Add the patches to the list depending upon whether they are the proper noise level
                if noise_level == NoiseLevel.LOW and std < low_noise_threshold:
                    x_filtered.append(x_patch)
                    y_filtered.append(y_patch)
                    stds.append(std)
                    continue
                elif noise_level == NoiseLevel.MEDIUM and low_noise_threshold < std < high_noise_threshold:
                    x_filtered.append(x_patch)
                    y_filtered.append(y_patch)
                    stds.append(std)
                    continue
                elif noise_level == NoiseLevel.HIGH and std > high_noise_threshold:
                    x_filtered.append(x_patch)
                    y_filtered.append(y_patch)
                    stds.append(std)
                    continue

            # Convert image patches and stds into numpy arrays
            x_filtered = np.array(x_filtered, dtype='uint8')
            y_filtered = np.array(y_filtered, dtype='uint8')
            stds = np.array(stds, dtype='float64')

            # Convert image patches and stds from (...,x,) to (...,x,1) shaped arrays
            x_filtered = x_filtered[..., np.newaxis]
            y_filtered = y_filtered[..., np.newaxis]
            stds = stds[..., np.newaxis]

            ''' Just logging
            # Plot the residual standard deviation
            image_utils.plot_standard_deviations(stds)
            '''

            # Assert that the last iteration has a full batch size
            assert len(x_original) % args.batch_size == 0, \
                logger.log(
                    'make sure the last iteration has a full batchsize, '
                    'this is important if you use batch normalization!')
            assert len(y_original) % args.batch_size == 0, \
                logger.log(
                    'make sure the last iteration has a full batchsize, '
                    'this is important if you use batch normalization!')

            # Standardize x and y to have a mean of 0 and standard deviation of 1
            # NOTE: x and y px values are centered at 0, meaning there are negative px values. We might have trouble
            # visualizing px that aren't either from [0, 255] or [0, 1], so just watch out for that
            x, x_orig_mean, x_orig_std = image_utils.standardize(x_filtered)
            y, y_orig_mean, y_orig_std = image_utils.standardize(x_filtered)

            ''' Just logging 
            logger.print_numpy_statistics(x, "x (standardized)")
            logger.print_numpy_statistics(y, "y (standardized)")
            '''

            '''Just for logging
            # Save the reversed standardization of x and y into variables
            x_reversed = image_utils.reverse_standardize(x, x_orig_mean, x_orig_std)
            y_reversed = image_utils.reverse_standardize(y, y_orig_mean, y_orig_std)
            '''

            # Get a list of indices, from 0 to the total number of training examples
            indices = list(range(x.shape[0]))

            # Make sure that x and y have the same number of training examples
            assert indices == list(range(y.shape[0])), logger.log('Make sure x and y are paired up properly! That is, x'
                                                                  'is a ClearImage, and y is a CoregisteredBlurryImage'
                                                                  'but that the two frames match eachother. ')

            # Increment the counter
            counter += 1

        # Iterate over the number of epochs
        for _ in range(num_epochs):

            # Shuffle the indices of the training examples
            np.random.shuffle(indices)

            # Iterate over the entire training set, skipping "batch_size" at a time
            for i in range(0, len(indices), batch_size):
                # Get the batch_x (clear) and batch_y (blurry)
                batch_x = x[indices[i:i + batch_size]]
                batch_y = y[indices[i:i + batch_size]]

                '''Just logging
                # Get equivalently indexed batches from x_original, x_reversed, y_original, and y_reversed
                batch_x_original = x_original[indices[i:i + batch_size]]
                batch_x_reversed = x_reversed[indices[i:i + batch_size]]
                batch_y_original = y_original[indices[i:i + batch_size]]
                batch_y_reversed = y_reversed[indices[i:i + batch_size]]

                # Show some images from this batch
                logger.show_images(images=[("batch_x[0]", batch_x[0]),
                                         ("batch_x_original[0]", batch_x_original[0]),
                                         ("batch_x_reversed[0]", batch_x_reversed[0]),
                                         ("batch_y[0]", batch_y[0]),
                                         ("batch_y_original[0]", batch_y_original[0]),
                                         ("batch_y_reversed[0]", batch_y_reversed[0])])
                '''

                # Finally, yield x and y, as this function is a generator
                yield batch_y, batch_x


def sum_squared_error(y_true, y_pred):
    """
    Returns sum-squared error between y_true and y_pred.
    This is the loss function for the network

    :param y_true: Target
    :type y_true: numpy array
    :param y_pred: Prediction
    :type y_pred: numpy array

    :return: Sum-Squared Error between the two
    :rtype: float
    """
    return K.sum(K.square(y_pred - y_true)) / 2


def original_callbacks():
    """
    Creates a list of callbacks for the Model Training process.
    This is a copy of the list of callbacks used for the original DnCNN paper

    :return: List of callbacks
    :rtype: list
    """

    # noinspection PyListCreation
    callbacks = []

    # Add checkpoints every <save_every> # of iterations
    callbacks.append(ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                     verbose=1, save_weights_only=False, save_freq=args.save_every))

    # Add the ability to log training information to <save_dir>/log.csv
    callbacks.append(CSVLogger(os.path.join(save_dir, 'log.csv'), append=True, separator=','))

    # Add a Learning Rate Scheduler to dynamically change the learning rate over time
    callbacks.append(LearningRateScheduler(lr_schedule))

    return callbacks


def new_callbacks():
    """
    Creates a list of callbacks for the Model Training process.
    This is the new list of callbacks used for MyDenoiser

    :return: List of callbacks
    :rtype: list
    """

    # noinspection PyListCreation
    callbacks = []

    # Add checkpoints every <save_every> # of iterations
    callbacks.append(ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                     verbose=1, save_weights_only=False, save_freq=args.save_every))

    # Add the ability to log training information to <save_dir>/log.csv
    callbacks.append(CSVLogger(os.path.join(save_dir, 'log.csv'), append=True, separator=','))

    # Add a Learning Rate Scheduler to dynamically change the learning rate over time
    callbacks.append(LearningRateScheduler(lr_schedule))

    # Add Early Stopping so that we stop training once val_loss stops decreasing after <patience> # of epochs
    callbacks.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3))

    return callbacks


def main():
    """
    Creates and trains the MyDenoiser Keras model.
    If no checkpoints exist, we will start from scratch.
    Otherwise, training will resume from previous checkpoints.

    :return: None
    """

    # Select the type of model to use
    if args.model == 'MyDnCNN':
        # Create a MyDnCNN model
        model = model_functions.MyDnCNN(depth=17, filters=64, image_channels=1, use_batchnorm=True)
    elif args.model == 'MyDenoiser1':
        # Create a MyDenoiser1 model
        model = model_functions.MyDenoiser1(image_channels=1, num_blocks=4)

    model.summary()

    # Load the last model
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = load_model(os.path.join(save_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)

    # Compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)

    # Train the model
    history = model.fit(my_train_datagen(batch_size=args.batch_size, data_dir=args.train_data),
                        steps_per_epoch=2000,
                        epochs=args.epoch,
                        initial_epoch=initial_epoch,
                        callbacks=new_callbacks())


if __name__ == '__main__':
    # Run the main function
    main()
