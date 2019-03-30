"""
2019/02/04

Hands on Machine Learning with scikit-learn and TensorFlow  -- Chapter 14

Example 2. Packing sequences
"""
import numpy as np
import tensorflow as tf


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# =========================================================================== #
# Packing sequences
# =========================================================================== #
n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
print('X: ', X)
_X = tf.transpose(X, perm=[1, 0, 2])
print('_X: ', _X)
X_seqs = tf.unstack(tf.transpose(X, [1, 0, 2]))
print('X_seqs: ', X_seqs)   # n_time_steps, ? mini_batch, 3 inputs each instance

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs_seqs, states = tf.nn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

print('outputs_seqs: ', outputs_seqs)
outputs = tf.transpose(tf.stack(outputs_seqs), perm=[0, 1, 2])
print('outputs: ', outputs)

init = tf.global_variables_initializer()


X_batch = np.array([
    # t = 0,    # t = 1
    [[0, 1, 2], [9, 8, 7]],     # instance 1
    [[3, 4, 5], [0, 0, 0]],     # instance 2
    [[6, 7, 8], [6, 5, 4]],     # instance 3
    [[9, 0, 1], [3, 2, 1]],     # instance 4
])

print(X_batch, X_batch.shape)

with tf.Session() as sess:
    init.run()
    output_vals = outputs.eval(feed_dict={X: X_batch})

print('output_vals: ', output_vals)
print('output_vals after transpose: ', np.transpose(output_vals, axes=[1, 0, 2])[1])


# =========================================================================== #
# Using dynamic_rnn()
# =========================================================================== #
reset_graph()

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()

X_batch = np.array([
        [[0, 1, 2], [9, 8, 7]],     # instance 1
        [[3, 4, 5], [0, 0, 0]],     # instance 2
        [[6, 7, 8], [6, 5, 4]],     # instance 3
        [[9, 0, 1], [3, 2, 1]],     # instance 4
    ])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

print('Using dynamic_rcnn(): ', outputs_val)


# =========================================================================== #
# Setting the sequence lengths
# =========================================================================== #
reset_graph()

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)

seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)


init = tf.global_variables_initializer()


X_batch = np.array([
    # step 0     step 1
    [[0, 1, 2], [9, 8, 7]],     # instance 1
    [[3, 4, 5], [0, 0, 0]],     # instance 2 (padded with zero vectors)
    [[6, 7, 8], [6, 5, 4]],     # instance 3
    [[9, 0, 1], [3, 2, 1]],     # instance 4
])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

print('In setting the sequence lengths:\n ', output_vals)
print('In setting the sequence lengths:\n ', states_val)

