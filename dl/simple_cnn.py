#!/usr/bin/env python
# coding=utf8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)

# Parameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 256

# Network Parameters
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# Store layers weight & bias
weights = {
    'conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'full1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'full2': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'conv1_bias': tf.Variable(tf.random_normal([32])),
    'conv2_bias': tf.Variable(tf.random_normal([64])),
    'full1_bias': tf.Variable(tf.random_normal([1024])),
    'full2_bias': tf.Variable(tf.random_normal([n_classes]))}


def conv2d(x, W, b, stride=1):
    h = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding="SAME")
    h = tf.nn.bias_add(h, b)
    return tf.nn.relu(h)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


def simple_conv_net(x, weights, biases, keep_probability):
    # convolutional layer1: 28*28*1 to 28*28*32
    conv1 = conv2d(x, weights['conv1'], biases['conv1_bias'])
    # max-pooling layer1: 28*28*32 to 14*14*32
    max_pool1 = maxpool2d(conv1, k=2)

    # convolutional layer2: 14*14*32 to 14*14*64
    conv2 = conv2d(max_pool1, weights['conv2'], biases['conv2_bias'])
    # max-pooling layer2: 14*14*64 to 7*7*64
    max_pool2 = maxpool2d(conv2, k=2)

    # reshape
    full1_input = tf.reshape(max_pool2, [-1, weights['full1'].get_shape().as_list()[0]])

    # fully connected layer 1
    fc1 = tf.add(tf.matmul(full1_input, weights['full1']), biases['full1_bias'])
    fc1 = tf.nn.relu(fc1)
    # apply drop out
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_probability)

    # fully connected layer 2
    fc2 = tf.add(tf.matmul(fc1, weights['full2']), biases['full2_bias'])
    return fc2


input_images = tf.placeholder(tf.float32, [None, 28, 28, 1])
labels = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

logits = simple_conv_net(input_images, weights, biases, keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    # initialize all variables
    session.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for batch in range(mnist_data.train.num_examples // batch_size):
            batch_images, batch_labels = mnist_data.train.next_batch(batch_size)
            _, loss_value = session.run([optimizer, loss], feed_dict={
                input_images: batch_images,
                labels: batch_labels,
                keep_prob: dropout,
            })
            valid_accuracy = session.run(accuracy, feed_dict={
                input_images: mnist_data.validation.images[:test_valid_size],
                labels: mnist_data.validation.labels[:test_valid_size],
                keep_prob: 1.0,
            })
            print("epoch %d, batch %d - Loss %f, validation accuracy: %f" % \
                  (epoch, batch, loss_value, valid_accuracy))

    test_accuracy = session.run(accuracy, feed_dict={
        input_images: mnist_data.test.images[:test_valid_size],
        labels: mnist_data.test.labels[:test_valid_size],
        keep_prob: 1.0,
    })
    print("test accuracy %f" % test_accuracy)
