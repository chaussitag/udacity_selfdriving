#!/usr/bin/env python
# coding=utf-8

# Load pickled data
import pickle
import numpy as np

# Initial Setup for Keras
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

from sklearn.preprocessing import LabelBinarizer

print("keras.__version__: ", keras.__version__)


def rgb_to_gray(rgb_image_batch):
    # Y = 0.2125 R + 0.7154 G + 0.0721 B
    gray_images = 0.2125 * rgb_image_batch[:, :, :, 0] + \
                  0.7154 * rgb_image_batch[:, :, :, 1] + \
                  0.0721 * rgb_image_batch[:, :, :, 2]
    gray_images = np.expand_dims(gray_images, axis=3)
    return gray_images.astype(np.float32)


def nomalize_images(images_data):
    # normalize each pixel value to [-1, 1]
    norm_data = (images_data - 128.0) / 128.0
    return norm_data.astype(np.float32)

with open('small_traffic_set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

epochs = 60
dropout_prob = 0.4
batch_size = 128
regularizer_factor = 0.1

# Build the Final Test Neural Network in Keras Here
model = Sequential()
model.add(Convolution2D(28, 5, 5, input_shape=(32, 32, 1),
                        W_regularizer=regularizers.l2(regularizer_factor),
                        b_regularizer=regularizers.l2(regularizer_factor)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout_prob))
model.add(Activation("relu"))

model.add(Flatten())

# model.add(Dense(64,
#                 W_regularizer=regularizers.l2(regularizer_factor),
#                 b_regularizer=regularizers.l2(regularizer_factor)))
# model.add(Dropout(dropout_prob))
# model.add(Activation("relu"))

model.add(Dense(32,
                W_regularizer=regularizers.l2(regularizer_factor),
                b_regularizer=regularizers.l2(regularizer_factor)))
model.add(Dropout(dropout_prob))
model.add(Activation("relu"))

model.add(Dense(5,
                W_regularizer=regularizers.l2(regularizer_factor),
                b_regularizer=regularizers.l2(regularizer_factor)))
model.add(Activation("softmax"))

# preprocess data
X_train = rgb_to_gray(X_train)
X_normalized = nomalize_images(X_train)

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=epochs, batch_size=batch_size, validation_split=0.2)

with open('small_traffic_set/small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

print("X_test.shape %s" % (X_test.shape, ))

# preprocess data
X_test = rgb_to_gray(X_test)
X_normalized_test = nomalize_images(X_test)
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

# Evaluate the test data in Keras Here
metrics = model.evaluate(X_test, y_one_hot_test)

for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
