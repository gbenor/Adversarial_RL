import os
from pathlib import Path

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.backend import categorical_crossentropy

''' Normalize input to the range of [0..1]
    Apart from assisting in the convergance of the training process, this
    will also make our lives easier during the adversarial attack process
'''
def normalize(x_train, x_test):
    x_train -= x_train.min()
    x_train /= x_train.max()
    x_test -= x_test.min()
    x_test /= x_test.max()

    return x_train, x_test


def extract_two_classes(img, labels, cls1, cls2):
    i = np.where((labels == cls1) | (labels == cls2))
    img = img[i]
    labels = labels[i]
    return img, labels

# Load and prepare the datasets for training
def load_data(cls1=None, cls2=None):
    num_classes = 10

    img_rows, img_cols, img_colors = 28, 28, 1
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    train_images, test_images = normalize(train_images, test_images)

    if (cls1 is not None) and (cls2 is not None):
        train_images, train_labels = extract_two_classes(train_images, train_labels, cls1, cls2)
        test_images, test_labels = extract_two_classes(test_images, test_labels, cls1, cls1)
        num_classes = 2

    # train_labels = keras.utils.to_categorical(train_labels, num_classes)
    # test_labels = keras.utils.to_categorical(test_labels, num_classes)
    return train_images, train_labels, test_images, test_labels

