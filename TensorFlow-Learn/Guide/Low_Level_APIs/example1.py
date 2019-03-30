from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Define the data
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
print(x)
print(y_true)

# Define the model
linear_model = tf.layers.Dense(units=1)
print(linear_model)

#
y_pred = linear_model(x)

# Define the loss
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(loss)

# Training:
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
print(train)

# Initialize:
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    print(sess.run(y_pred))
    t = tf.Print(x, [x])
    result = t + 1
    result.eval()

writer = tf.summary.FileWriter('/tmp/Low_Level_APIs/')
writer.add_graph(tf.get_default_graph())
writer.flush()

