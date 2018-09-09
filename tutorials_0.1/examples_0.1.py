import os
import random
import numpy as np
import skimage.data
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray


DATA_DIRECTORY = './data/'
TRAIN_DIRECTORY = DATA_DIRECTORY + 'Training'
TEST_DIRECTORY = DATA_DIRECTORY + 'Testing'


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith('.ppm')]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


def statistic(images, labels):
    plt.hist(labels, 62)
    plt.show()
    visualize(images, labels)
    visualize1(images, labels)


def visualize(images, labels):
    # Determine the (random) indexes of the images that you want to see
    traffic_signs = [300, 2250, 3650, 4000]

    # Fill out the subplots with the random images that you defined
    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(images[traffic_signs[i]])
        plt.subplots_adjust(wspace=0.5)
        print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                      images[traffic_signs[i]].min(),
                                                      images[traffic_signs[i]].max()))
    plt.show()


def visualize1(images, labels):
    unique_labels = set(labels)     # Get the unique labels
    plt.figure(figsize=(15, 15))    # Initialize the figure
    i = 1   # Set a counter
    # For each unique label,
    for label in unique_labels:
        # You pick the first image for each label
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)    # Define 64 subplots
        plt.axis('off')         # Don't include axes
        # Add a title to each subplot
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1                  # Add 1 to the counter
        # And you plot this first image
        plt.imshow(image)
    plt.show()


def rescale_convert(images, labels):
    # Rescale the images in the `images` array
    images28 = [transform.resize(image, (28, 28), mode='constant') for image in images]
    images28 = np.array(images28)   # Convert `images28` to an array
    images28 = rgb2gray(images28)   # Convert `images28` to grayscale
    traffic_signs = [300, 2250, 3650, 4000]
    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images28[traffic_signs[i]], cmap='gray')
        plt.subplots_adjust(wspace=0.5)
    plt.show()
    return images28, labels


def network_model_train(images28, labels):
    # Initialize placeholder
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
    y = tf.placeholder(dtype=tf.int32, shape=[None])
    # Flatten the input data
    images_flat = tf.contrib.layers.flatten(x)
    # Fully connected layer
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    # Define loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                         logits=logits))
    # Define an optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # Convert logits to label indexes
    correct_pred = tf.argmax(logits, 1)
    # Define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("images_flat: ", images_flat)         # Tensor("Flatten/flatten/Reshape:0", shape=(?, 784), dtype=float32)
    print("logits: ", logits)                   # Tensor("fully_connected/Relu:0", shape=(?, 62), dtype=float32)
    print("loss: ", loss)                       # Tensor("Mean:0", shape=(), dtype=float32)
    print("predicted_labels: ", correct_pred)   # Tensor("ArgMax:0", shape=(?,), dtype=int64)

    tf.set_random_seed(1234)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        # print('EPOCH ', i)
        _, accuracy_val, loss_val = sess.run([train_op, accuracy, loss],
                                             feed_dict={x: images28,
                                                        y: labels})
        if i % 10 == 0:
            print('Loss: ', loss_val)
        # print('Done with epoch')

    # Pick 10 random images
    sample_indexes = random.sample(range(len(images28)), 10)
    sample_images = [images28[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]
    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
    # Print the real and predicted labels
    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        print('###', i)
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        # color = 'green' if np.logical_and(truth, prediction) else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(sample_images[i], cmap="gray")
    plt.show()
    sess.close()


if __name__ == '__main__':
    images, labels = load_data(TRAIN_DIRECTORY)
    statistic(images, labels)
    images28, labels = rescale_convert(images, labels)
    network_model_train(images28, labels)

