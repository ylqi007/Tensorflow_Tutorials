"""
http://www.lunarnai.cn/2018/03/28/tensorflow_shape/
"""

import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.int32, shape=[5], name="x")   # The shape of x is defined.
y = tf.placeholder(tf.int32, shape=[None])          # Only the rank of y is defined.

print("static shape of x: ", x.get_shape(), type(x.get_shape()), x.get_shape().as_list())
print("static shape of y: ", y.get_shape(), type(y.get_shape()), y.get_shape().as_list())
print("dynamic shape of x: ", tf.shape(x, name="shape_of_x"), type(tf.shape(x)))
print("dynamic shape of y: ", tf.shape(y, name="shape_of_y"))


with tf.Session() as sess:
    x_shape, y_shape = sess.run([tf.shape(x), tf.shape(y)], feed_dict={x: [1, 2, 3, 4, 5], y: [3, 2, 1]})
    print(x_shape)
    print(y_shape)
