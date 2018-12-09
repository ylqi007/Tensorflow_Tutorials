from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow.contrib import autograph

import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()


# def square_if_positive(x):
#     if x > 0:
#         x = x * x
#     else:
#         x = 0.0
#     return x
#
#
# print(autograph.to_code(square_if_positive))
# print('Eager results: %2.2f, %2.2f' % (square_if_positive(tf.constant(9.0)), square_if_positive(tf.constant(-9.0))))
# tf_square_if_positive = autograph.to_graph(square_if_positive)
#
# with tf.Graph().as_default():
#     g_out1 = tf_square_if_positive(tf.constant(9.0))
#     g_out2 = tf_square_if_positive(tf.constant(-9.0))
#     with tf.Session() as sess:
#         print('Graph results: %2.2f, %2.2f\n' % (sess.run(g_out1), sess.run(g_out2)))
#
#         writer = tf.summary.FileWriter('/tmp/TensorFlow_Guild/')    # tensorboard --logdir /tmp/TensorFlow_Guide/
#         writer.add_graph(tf.get_default_graph())


# Continue in a loop
# def sum_even(items):
#     s = 0
#     for c in items:
#         if c % 2 > 0:
#             continue
#         s += c
#     return s
#
#
# print(autograph.to_code(sum_even))
# print('Eager result: %d' % sum_even(tf.constant([10, 12, 15, 20])))
#
# tf_sum_even = autograph.to_graph(sum_even)
# with tf.Graph().as_default(), tf.Session() as sess:
#     print('Graph result: %d\n\n' % sess.run(tf_sum_even(tf.constant([10, 12, 15, 20]))))


# =========================================================================== #
# Decorator
# =========================================================================== #
@autograph.convert()
def fizzbuzz(i, n):
    while i < n:
        msg = ''
        if i % 3 == 0:
            msg += 'Fizz'
        if i % 5 == 0:
            msg += 'Buzz'
        if msg == '':
            msg = tf.as_string(i)
        print(msg)
        i += 1
    return i


with tf.Graph().as_default():
    final_i = fizzbuzz(tf.constant(10), tf.constant(16))
    # The result works like a regular op: takes tensors in, returns tensors.
    # You can inspect the graph using tf.get_default_graph().as_graph_def()
    with tf.Session() as sess:
        sess.run(final_i)


# =========================================================================== #
# Assert
# =========================================================================== #
@autograph.convert()
def inverse(x):
    assert x != 0.0, 'Do not pass zero!'
    return 1.0 / x


with tf.Graph().as_default(), tf.Session() as sess:
    try:
        print(sess.run(inverse(tf.constant(0.0))))
    except tf.errors.InvalidArgumentError as e:
        print('Got error message:\n    %s' % e.message)


# =========================================================================== #
# List
# =========================================================================== #


