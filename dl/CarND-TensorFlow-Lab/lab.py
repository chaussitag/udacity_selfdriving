#!/usr/bin/env python
# coding=utf8
import hashlib
import os
from urllib.request import urlretrieve
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from zipfile import ZipFile
import pickle
import math
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt


def download_if_necessary(url, file, md5):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    :param md5: md5 sum of the content of file if existed
    """
    if not os.path.isfile(file) or hashlib.md5(open(file, 'rb').read()).hexdigest() != md5:
        if os.path.isfile(file):
            os.remove(file)
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')


def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    feature_list = []
    label_list = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                feature_list.append(feature)
                label_list.append(label)
    return np.array(feature_list), np.array(label_list)


def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    gray_min = 0.0
    gray_max = 255.0
    target_min = 0.1
    target_max = 0.9
    return target_min + (image_data - gray_min) * (target_max - target_min) / (gray_max - gray_min)


pickle_file = 'notMNIST.pickle'


def load_data():
    try:
        # Reload the data from saved pickle file
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
            train_features_ = pickle_data['train_dataset']
            train_labels_ = pickle_data['train_labels']
            valid_features_ = pickle_data['valid_dataset']
            valid_labels_ = pickle_data['valid_labels']
            test_features_ = pickle_data['test_dataset']
            test_labels_ = pickle_data['test_labels']
            del pickle_data  # Free up memory
            return train_features_, train_labels_, valid_features_, valid_labels_, test_features_, test_labels_
    except IOError:
        print("failed to reload data from %s, load it from begining" % pickle_file)

    # Download the training and test dataset.
    download_if_necessary('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip',
                          'notMNIST_train.zip', 'c8673b3f28f489e9cdf3a3d74e2ac8fa')
    download_if_necessary('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip',
                          'notMNIST_test.zip', '5d3c7e653e63471c88df796156a9dfa9')
    # Make sure the files aren't corrupted
    assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa', \
        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
    assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9', \
        'notMNIST_test.zip file is corrupted.  Remove the file and try again.'

    # Get the features and labels from the zip files
    train_features_, train_labels_ = uncompress_features_labels('notMNIST_train.zip')
    test_features_, test_labels_ = uncompress_features_labels('notMNIST_test.zip')

    # Limit the amount of data to work with a docker container
    docker_size_limit = 150000
    train_features_, train_labels_ = resample(train_features_, train_labels_, n_samples=docker_size_limit)

    # normalize the data
    train_features_ = normalize_grayscale(train_features_)
    test_features_ = normalize_grayscale(test_features_)

    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(train_labels_)
    # one-hor encode, and change to float32,
    # so it can be multiplied against the features in TensorFlow, which are float32
    train_labels_ = encoder.transform(train_labels_).astype(np.float32)
    test_labels_ = encoder.transform(test_labels_).astype(np.float32)

    # Get randomized datasets for training and validation
    train_features_, valid_features_, train_labels_, valid_labels_ = train_test_split(
        train_features_,
        train_labels_,
        test_size=0.05,
        random_state=832289)

    # Save the data for easy access
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open('notMNIST.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': train_features_,
                        'train_labels': train_labels_,
                        'valid_dataset': valid_features_,
                        'valid_labels': valid_labels_,
                        'test_dataset': test_features_,
                        'test_labels': test_labels_,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
                print('Data cached in pickle file.')
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        return train_features_, train_labels_, valid_features_, valid_labels_, test_features_, test_labels_


train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_data()

features_count = 784
labels_count = 10

features = tf.placeholder(tf.float32, [None, features_count])
labels = tf.placeholder(tf.float32, [None, labels_count])

weights = tf.Variable(tf.random_normal([features_count, labels_count]))
biases = tf.Variable(tf.zeros([labels_count]))

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# Linear Function WX + b
logits = tf.matmul(features, weights) + biases

prediction = tf.nn.softmax(logits)

# Cross entropy
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)

# Training loss
loss = tf.reduce_mean(cross_entropy)

# Create an operation that initializes all variables
init = tf.global_variables_initializer()

# Test Cases
with tf.Session() as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    session.run(loss, feed_dict=test_feed_dict)
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

epochs = 5
batch_size = 100
learning_rate = 0.2

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features) / batch_size))

    for epoch_i in range(epochs):

        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i * batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

plt.ion()
loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))

epochs = 5
batch_size = 100
learning_rate = 0.2

# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features) / batch_size))

    for epoch_i in range(epochs):

        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i * batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
        print("epoch - %d: test accuracy %f" % (epoch_i, test_accuracy))

assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))