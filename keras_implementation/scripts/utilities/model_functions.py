"""Contains functions for creating Neural Nets using the Keras Function API"""

from tensorflow.keras.layers import Input, Conv2D, Conv3D, BatchNormalization, Activation, Subtract

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
import glob
import os
import re


def findLastCheckpoint(save_dir: str):
    """
    Finds the most epoch number from a directory of saved models

    :param save_dir: The directory where the model_*.hdf5 files are located
    :return: initial_epoch: The most recent epoch number
    """
    file_list = glob.glob(os.path.join(save_dir, 'model_*.hdf5'))  # get name list of all .hdf5 files
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def My3dDenoiser(depth, num_filters=64, use_batchnorm=True):
    """
    Complete implementation of My3dDenoiser, a 3D residual CNN using TensorFlow.

    Parameters
    ----------
    depth: The total number of layers for the network
    num_filters: The total number of convolutional kernels in each convolutional layer
        of the network
    use_batchnorm: Whether or not the layers of the network should use batch normalization

    Returns
    -------
    A My3dDenoiser model, defined using TensorFlow 2 API
    """
    layer_index = 2

    # Layer 0 - The input layer (An entire brain scan)
    input_layer = Input(shape=(None, None, None), name='Input')

    # Layer 1 - Convolutional Layer + ReLU activation
    x = Conv3D(filters=num_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_initializer='Orthogonal',
               padding='same', name=f'Conv{layer_index}')(input_layer)
    x = Activation('relu', name=f'ReLU{layer_index}')(x)

    # Layer 2 through Layer (depth - 1)
    for i in range(depth-2):
        # Layer N - Convolutional Layer + (Optionally) BatchNorm + ReLU Activation
        x = Conv3D(filters=num_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_initializer='Orthogonal',
                   padding='same', use_bias=False, name=f'Conv{layer_index}')(x)
        if use_batchnorm:
            x = BatchNormalization(axis=4, momentum=0.0, epsilon=0.0001, name=f'BatchNorm{layer_index}')(x)
        x = Activation('relu', name=f'ReLU{layer_index}')(x)
        layer_index += 1

    # Final Layer
    x = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_initializer='Orthogonal', padding='same',
               use_bias=False, name=f'Conv{layer_index}')(x)
    x = Subtract(name=f'Subtract{layer_index}')([input_layer, x])

    # Finally, define the model
    return Model(inputs=input_layer, outputs=x)



def MyDnCNN(depth, filters=64, image_channels=1, use_batchnorm=True):
    """
    Complete implementation of MyDnCNN, a residual network using the Keras API.
    MyDnCNN is originally derived from DnCNN, but with some changes and
    reorganization

    :param depth: The total number of layers for the network, colloquially referred to
                    as the "depth" of the network
    :param filters: The total number of convolutional kernels in each convolutional
                    layer of the network
    :param image_channels: The number of dimensions of the input images, i.e.
                            image_channels=1 for grayscale images, or image_channels=3
                            for RGB images.
    :param use_batchnorm: Whether or not the layers of the network should use batch
                            normalization
    :return: A MyDnCNN model, defined using the Keras API
    """

    # Initialize counter to keep track of current layer
    layer_index = 0

    # Define Layer 0 -- The input layer, and increment layer_index
    input_layer = Input(shape=(None, None, image_channels), name='Input' + str(layer_index))
    layer_index += 1

    # Define Layer 1 -- Convolutional Layer + ReLU activation function, and increment layer_index
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='Conv' + str(layer_index))(input_layer)
    layer_index += 1
    x = Activation('relu', name='ReLU' + str(layer_index))(x)

    # Iterate through the rest of the (depth - 2) layers -- Convolutional Layer + (Maybe) BatchNorm layer + ReLU
    for i in range(depth - 2):

        # Define Convolutional Layer
        layer_index += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='Conv' + str(layer_index))(x)

        # (Optionally) Define BatchNormalization layer
        if use_batchnorm:
            layer_index += 1
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='BatchNorm' + str(layer_index))(x)

        # Define ReLU Activation Layer
        layer_index += 1
        x = Activation('relu', name='ReLU' + str(layer_index))(x)

    # Define last layer -- Convolutional Layer and Subtraction Layer (input - noise)
    layer_index += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same',
               use_bias=False, name='Conv' + str(layer_index))(x)
    layer_index += 1
    x = Subtract(name='Subtract' + str(layer_index))([input_layer, x])

    # Finally, define the model
    model = Model(inputs=input_layer, outputs=x)

    return model


def MyDenoiser(image_channels=1, num_blocks=4):
    """
    Complete implementation of MyDenoiser, a residual network using the Keras API.

    :param image_channels: The number of dimensions of the input images, i.e.
                            image_channels=1 for grayscale images, or image_channels=3
                            for RGB images.
    :type image_channels: int
    :param num_blocks: The number of Repeatable Denoising Inception-Residual Blocks in the network
    :type num_blocks: int

    :return: A MyDenoiser model, defined using the Keras API
    """

    '''Initial Setup'''
    # Initialize counter to keep track of current layer
    layer_index = 0

    # Set the number of filters (The number of output filters in the convolution)
    filters = 64

    '''Layer Definitions'''
    # Define Layer 0 -- Input Layer, and increment layer_index
    input_layer = Input(shape=(None, None, image_channels), name='Input' + str(layer_index))
    layer_index += 1

    # Define Layer 1 -- Convolutional Layer (64 7x7 kernels) + ReLU activation function, and increment layer_index
    x = Conv2D(filters=filters, kernel_size=(7, 7), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='Conv' + str(layer_index))(input_layer)
    layer_index += 1
    x = Activation('relu', name='ReLU' + str(layer_index))(x)

    # Define Layer 2 -- Convolutional Layer (64 3x3 kernels) + ReLU activation function, and increment layer_index
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='Conv' + str(layer_index))(x)
    layer_index += 1
    x = Activation('relu', name='ReLU' + str(layer_index))(x)

    # Iterate through the rest of the (num_blocks) layers -- Repeatable Denoising Inception-Residual Block
    for i in range(num_blocks):

        # Define Convolutional Layer
        layer_index += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='Conv' + str(layer_index))(x)

        # Define BatchNormalization layer
        layer_index += 1
        x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='BatchNorm' + str(layer_index))(x)

        # Define ReLU Activation Layer
        layer_index += 1
        x = Activation('relu', name='ReLU' + str(layer_index))(x)

    # Define last layer -- Convolutional Layer and Subtraction Layer (input - noise)
    layer_index += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same',
               use_bias=False, name='Conv' + str(layer_index))(x)
    layer_index += 1
    x = Subtract(name='Subtract' + str(layer_index))([input_layer, x])

    # Finally, define the model
    model = Model(inputs=input_layer, outputs=x)

    return model