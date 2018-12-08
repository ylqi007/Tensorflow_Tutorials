"""
https://www.tensorflow.org/guide/variables
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# =========================================================================== #
# 1. Creating a Variable
# =========================================================================== #
my_variable = tf.get_variable("my_variable", [1, 2, 3])     # shape=(1, 2, 3) dtype=float32_ref
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23, 42]))
# print(my_variable)
# print(my_int_variable)
# print(other_variable)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(my_variable.eval())
#     print(my_int_variable.eval())
#     print(other_variable.eval())


# =========================================================================== #
# 1.1 Variable collections
# =========================================================================== #
my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
my_non_trainable = tf.get_variable("my_non_trainable", shape=(), trainable=False)
print(my_local)
print(my_non_trainable)

# =========================================================================== #
# 1.2 Device placement
# =========================================================================== #
# with tf.Session() as sess:
#     with tf.device("/device:GPU:0"):
#         v = tf.get_variable("v", [1])
#     print(v)

cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
    v = tf.get_variable("v", shape=[20, 20])    # this variable is placed
                                                # in the parameter server
                                                # by the replica_device_setter


# =========================================================================== #
# 4. Sharing variables
#   1) Explicitly passing `tf.Variable` objects around.
#   2) Implicitly wrapping `tf.Variable` objects within `tf.variable_scope` objects.
# =========================================================================== #
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights"
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases"
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])
# x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
# x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])      # This fails.


def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])


with tf.variable_scope("model"):
    output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
    output2 = my_image_filter(input2)


with tf.variable_scope("model1") as scope:
    output11 = my_image_filter(input1)
    scope.reuse_variables()
    output22 = my_image_filter(input2)


with tf.variable_scope("model2") as scope:
    output111 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
    output222 = my_image_filter(input2)
