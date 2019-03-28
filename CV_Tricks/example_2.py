"""
https://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow/tensorflow-tutorial/
"""

import tensorflow as tf
import numpy as np


# Reduce mean
print("######### test for reduce_mean ##########")
b = tf.Variable([[10, 20, 30], [30, 40, 50]], name='t')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(tf.reduce_mean(b, axis=0)))
    print(sess.run(tf.reduce_mean(b, axis=1)))


# ArgMax: Gets you the maximum value from a tensor along the spcified axis.
print("######### test for argmax ##########")
a = [[0.1, 0.2, 3],
     [20, 3, 2]]
b1 = tf.Variable(a, name='b1')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(tf.argmax(b1, axis=0)))
    print(sess.run(tf.argmax(b1, axis=1)))


###############################################################################
# Linear Regression Exercise
# In linear regression, you get a lot of data-points and try to fit them on a
# straight line.
###############################################################################
# 1. Creating training data: trainX has values between -1 and 1, and trainY has
# 4 times trainX and some randomness.
trainX = np.linspace(-1, 1, 101)
trainY = 4 * trainX + np.random.randn(*trainX.shape) * 0.33
# 2. Creating Placeholder:
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 3.1 Modeling:
w = tf.Variable(0.0, name='weights')
y_pred = tf.multiply(X, w)
# 3.2 Cost:
cost = tf.pow(Y - y_pred, 2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# 4. Training:
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))



