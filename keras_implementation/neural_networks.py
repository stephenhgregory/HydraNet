#!usr/bin/env python

"""
Contains class definitions and implementations of neural networks
"""

import argparse
import re
import os, glob, datetime
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract, MaxPooling2D
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import keras_implementation.utilities.data_generator
import keras.backend as K

# Fixes a TensorFlow bug relating to memory issues ######################
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print(f'The following line threw an exception: tf.config.experimental.set_memory_growth(physical_devices[0], True)')
    pass
##########################################################################

# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=300, type=int, help='number of train epochs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epochs')
args = parser.parse_args()


class MyDenoiser(Model):

    def __init__(self, classes, filters=64, kernel_initializer='Orthogonal', channel_dimensions=-1):
        # call the parent constructor
        super(MyDenoiser, self).__init__()
        # initialize the layers in the first (CONV => RELU => BatchNorm) * 2 => POOL
        # layer set
        self.conv1A = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='Orthogonal', padding="same")
        self.act1A = Activation("relu")
        self.bn1A = BatchNormalization(axis=channel_dimensions)
        self.conv1B = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='Orthogonal', padding="same")
        self.act1B = Activation("relu")
        self.bn1B = BatchNormalization(axis=channel_dimensions)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        # initialize the layers in the second (CONV => RELU => BatchNorm) * 2 => POOL
        # layer set
        self.conv2A = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='Orthogonal', padding="same")
        self.act2A = Activation("relu")
        self.bn2A = BatchNormalization(axis=channel_dimensions)
        self.conv2B = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='Orthogonal', padding="same")
        self.act2B = Activation("relu")
        self.bn2B = BatchNormalization(axis=channel_dimensions)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        # initialize the layers in our fully-connected layer set
        self.flatten = Flatten()
        self.dense3 = Dense(512)
        self.act3 = Activation("relu")
        self.bn3 = BatchNormalization()
        self.do3 = Dropout(0.5)
        # initialize the layers in the softmax classifier layer set
        self.dense4 = Dense(classes)
        self.softmax = Activation("softmax")

    def call(self, inputs, training=None, mask=None):
        pass

