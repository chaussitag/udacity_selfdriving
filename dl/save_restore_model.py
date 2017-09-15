#!/usr/bin/env python
# coding=utf8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

# Remove previous Tensors and Operations
tf.reset_default_graph()

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, shape=[None, n_input])
labels = tf.placeholder(tf.float32, shape=[None, n_classes])

# full connected layer 1
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# logits
logits = tf.add(tf.matmul(features, weights), bias)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# train to minimize the loss
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

model_save_path = "./simple_minist_model.ckpt"
batch_size = 128
epoches = 100
with tf.Session() as session:
    # must run this operation to initialize all variables
    session.run(tf.global_variables_initializer())

    for epoch in range(epoches):
        num_of_batches = math.ceil(mnist_data.train.num_examples / batch_size)

        for i in range(num_of_batches):
            batch_features, batch_labels = mnist_data.train.next_batch(batch_size)
            _, loss_value = session.run(
                [train_step, loss],
                feed_dict={
                    features: batch_features,
                    labels: batch_labels
                })
            if i % 100 == 0:
                print("epoch %d batch %d: loss %f" % (epoch, i, loss_value))

        # print status for every 10 epoches
        if epoch % 10 == 0:
            accuracy_value = session.run(
                accuracy,
                feed_dict={
                    features: mnist_data.validation.images,
                    labels: mnist_data.validation.labels
                })
            print("Epoch: {:<3} - Validation Accuracy: {}".format(epoch, accuracy_value))

    # save the trained model
    saver = tf.train.Saver()
    saver.save(session, model_save_path)
    print("Trained model saved to %s" % model_save_path)

with tf.Session() as session:
    # load the saved model and use it to predict
    saver = tf.train.Saver()
    saver.restore(session, model_save_path)
    test_accuracy = session.run(
        accuracy,
        feed_dict={
            features: mnist_data.test.images,
            labels: mnist_data.test.labels
        }
    )
    print("Test Accuracy: %f" % test_accuracy)