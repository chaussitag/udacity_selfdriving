import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import time
import matplotlib.pyplot as plt
import numpy as np

# the train.p is downloaded from https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p
# Load traffic signs data.
train_data_path = "./train.p"
with open(train_data_path, "rb") as f:
    train_data = pickle.load(f)
images, labels = train_data['features'], train_data['labels']

# Split data into training and validation sets.
train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.05)

train_image_shape = train_images.shape[1:4]
# Define placeholders and resize operation.
input_images = tf.placeholder(tf.float32, shape=[None] + list(train_image_shape))
resized_images = tf.image.resize_images(input_images, (227, 227))

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_images, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
fc7_dim = fc7.get_shape().as_list()[-1]
n_classes = 43

fc8_w = tf.Variable(tf.truncated_normal([fc7_dim, n_classes], stddev=0.01))
fc8_b = tf.Variable(tf.zeros(n_classes))
logits = tf.nn.xw_plus_b(fc7, fc8_w, fc8_b)

# Define loss, training, accuracy operations.
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 256

labels = tf.placeholder(tf.int32, None)
one_hot_labels = tf.one_hot(labels, n_classes)

loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_tensor)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy_tensor = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(test_images, test_labels, sess):
    total_loss_ = 0.0
    total_accuracy_ = 0.0
    num_images = len(test_images)
    for offset_ in range(0, num_images, BATCH_SIZE):
        end_ = offset_ + BATCH_SIZE
        batch_images_ = test_images[offset_:end_]
        batch_labels_ = test_labels[offset_:end_]
        loss_, accuracy_ = sess.run([loss_tensor, accuracy_tensor],
                                     feed_dict={
                                         input_images: batch_images_,
                                         labels: batch_labels_
                                     })
        total_loss_ += (loss_ * len(batch_images_))
        total_accuracy_ += (accuracy_ * len(batch_images_))

    return total_loss_ / num_images, total_accuracy_ / num_images

# Train and evaluate the feature extraction model.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
validation_losses = []
validation_accuracies = []
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    total_samples = train_images.shape[0]
    for epoch in range(0, EPOCHS):
        train_images, train_labels = shuffle(train_images, train_labels)
        t_start = time.time()
        for offset in range(0, total_samples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_images, batch_labels = train_images[offset:end], train_labels[offset:end]
            session.run(train_op, feed_dict={
                input_images: batch_images,
                labels: batch_labels
            })
        valid_loss, valid_accuracy = evaluate(valid_images, valid_labels, session)
        validation_losses.append(valid_loss)
        validation_accuracies.append(valid_accuracy)
        print("Epoch", epoch + 1)
        print("  Time(train and validation): %.3f seconds" % (time.time() - t_start))
        print("  Validation Loss =", valid_loss)
        print("  Validation Accuracy =", valid_accuracy)
        print("")

fig = plt.gcf()
axes1 = fig.add_subplot(1, 2, 1)
axes1.set_title("validation loss for epochs")
axes1.set_xlabel("epochs")
axes1.set_ylabel("loss")
axes1.set_xticks(range(0, EPOCHS))
axes1.set_yticks(np.arange(np.min(validation_losses), np.max(validation_losses), 0.01))
axes1.grid(True)
axes1.plot(range(0, EPOCHS), validation_losses)

axes2 = fig.add_subplot(1, 2, 2)
axes2.set_title("validation accuracy for epochs")
axes2.set_xlabel("epochs")
axes2.set_ylabel("accuracy")
axes2.set_xticks(range(0, EPOCHS))
axes2.set_yticks(np.arange(np.min(validation_accuracies), np.max(validation_accuracies), 0.005))
axes2.grid(True)
axes2.plot(range(0, EPOCHS), validation_accuracies)

plt.show()

