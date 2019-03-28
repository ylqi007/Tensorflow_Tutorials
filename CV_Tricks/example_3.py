"""
https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/

Types of layers:
1. All the neurons in one layer do similar kind of mathematical operations
and that's how that a layer gets its name.
    * Convolutional Layer: (N - F + 2*P) / S + 1
    * Pooling Layer: To reduce the spatial size.
    * Fully Connected Layer:
2. Understanding Training process: Deep neural networks are nothing but mathematical
models of intelligence which to a certain extent mimic human brains.
    * The Architecture of the network:
        - How do you arrange layers?
        - Which layers to use?
        - How many neurals to use in each layer etc?
        - Standard architectures: AlexNet, GoogleNet, InceptionResnet, VGG etc.
    * Correct weights/parameters:
        - The objective of the training is to get the best possible values of the all
            these parameters which solve the problem reliably.
        - Backward propagation: optimizers
        - Typically cost is defined in such a way that: as the cost is reduced, the
            accuracy of the network increases.
            * One of the simple one is mean root square cost.
        - Inference or prediction
        - batch-size * iteration = num of all images  ==>  one epoch
        -
"""

import os
import tensorflow as tf
from tensorflow import set_random_seed
from numpy.random import seed

from CV_Tricks import dataset

# Adding seed so that random initialization is consistent.
seed(1)
set_random_seed(2)

# Batch size
batch_size = 32

# Prepare input data
classes = os.listdir('training_data')       # ['dogs', 'cats']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path = 'training_data'

# Load all the training and validation images and labels into memory using openCV
# and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


session = tf.Session()

# training data
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


# c). Creating network layers:
# c.1). Building convolution layer in TensorFlow: tf.nn.conv2d
# filter = [filter_size filter_size num_input_channels num_filters]
# padding=SAME means we shall 0 pad the input such a way that the output x, y dimensions
#   are same as that of input.
# After convolution, we add the biases of that neuron, which are also learnable/trainable
def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # Define the weights that will be trained using create_weights function:
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # Define the biases using the create_biases function.
    biases = create_biases(num_filters)

    # Create the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # Using max-pooling
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')      # The output is exactly half of input.
    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


# c.2) Flattening layer:
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


# c.3) Fully connected layer:
def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since,
    # these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# c.5) Network design:
layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,       # 3
                                         conv_filter_size=filter_size_conv1,    # 3
                                         num_filters=num_filters_conv1)         # 32

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,  # 32
                                         conv_filter_size=filter_size_conv2,    # 32
                                         num_filters=num_filters_conv2)         # 32

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,  # 32
                                         conv_filter_size=filter_size_conv3,    # 64
                                         num_filters=num_filters_conv3)         # 64

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),      # ????
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

# c.6) Predictions:
y_pred = tf.nn.softmax(layer_fc2, name='y_pred')    # y_pred contains the predicted probability of each class for images
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# c.7) Optimization:
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0:3} - Training Accuracy: {1:>6.1%}, " \
          "Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0
saver = tf.train.Saver()


def train(num_iterations):
    global total_iterations
    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './dogs_cats_model/dogs-cats-model')

    total_iterations += num_iterations


train(num_iterations=3000)
