"""
2018-12-05 From MoFan
"""

import tensorflow as tf
import numpy as np


# Create input data.
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Create tensorflow structure -- Start #
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# Create tensorflow structure -- End #

with tf.Session() as sess:
    sess.run(init)  # Very Important.
    for step in range(200):
        sess.run(train)
        if step%20 == 0:
            print(step, sess.run(Weights), sess.run(biases))

