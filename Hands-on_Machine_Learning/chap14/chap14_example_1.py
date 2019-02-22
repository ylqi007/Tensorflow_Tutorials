"""
2019/02/04

Hands on Machine Learning with scikit-learn and TensorFlow  -- Chapter 11

Example 1. Basic RNNs in TensorFlow.
"""
# To support both python2 and python3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Common imports
import numpy as np
import os
import tensorflow as tf

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID, fig_id + '.png')
    print('Saving figure: ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# Basic RNNs -- Manual RNN
reset_graph()

n_inputs = 3    # Input vectors of size 3 at each time step.
n_neurons = 5   # A layer of five recurrent neurons.

X0 = tf.placeholder(tf.float32, [None, n_inputs])   # [None, 3]
X1 = tf.placeholder(tf.float32, [None, n_inputs])   # [None, 3]

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))   # [3, 5]
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))  # [5, 5]
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)                     # [None, 5]
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b) # [None, 5] + [None, 5] + [None, 5] = [None, 5]

init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])   # at t = 0, shape=(4, 3)
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])   # at t = 1, shape=(4, 3)

with tf.Session() as sess:
    init.run()
    # print('X0:\n', sess.run([X0], feed_dict={X0: X0_batch}))
    # print('b:\n', b.eval())
    # print('Wx:\n', Wx.eval())
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

print('Y0_val:\n', Y0_val)  # shape=(4, 5), 4 instances and each instance corresponding to a output of shape (1, 5)
print('Y1_val:\n', Y1_val)  # shape=(4, 5)

# =========================================================================== #
# static_rnn()
# The following code creates the exact same model as the previous one: static_rnn()
# =========================================================================== #
reset_graph()

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

print('basic_cell: ', basic_cell)
print('output_seqs: ', output_seqs)

Y0, Y1 = output_seqs
print('Y0: ', Y0)
print('Y1: ', Y1)
print('state: ', states)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val, states = sess.run([Y0, Y1, states], feed_dict={X0: X0_batch, X1: X1_batch})

print('Y0: ', Y0_val)
print('Y1: ', Y1_val)
print('states: ', states)