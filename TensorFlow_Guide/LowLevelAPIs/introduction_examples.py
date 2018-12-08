""" Introduction
https://www.tensorflow.org/guide/low_level_intro
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# a = tf.constant(3.0, dtype=tf.float32)
# b = tf.constant(4.0)    # Also tf.float32 implicitly
# c = tf.constant(5.0, name="my_Const")    # Also tf.float32 implicitly
# total = a + b + c
# print(a)
# print(b)
# print(c)
# print(total)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(a)
#     print(b.eval())
#     print(c)
#     print(total.eval())
#
#     writer = tf.summary.FileWriter('.')
#     writer.add_graph(tf.get_default_graph())


# =========================================================================== #
# Datasets
# =========================================================================== #
my_data = [[0, 1, ], [2, 3, ], [4, 5, ], [6, 7, ]]
# print(my_data)
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

# with tf.Session() as sess:
#     while True:
#         try:
#             print(sess.run(next_item))
#         except tf.errors.OutOfRangeError:
#             break


r = tf.random_normal([10, 3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     while True:
#         try:
#             print(sess.run(next_row))
#         except tf.errors.OutOfRangeError:
#             break


# =========================================================================== #
# Layers
# =========================================================================== #
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


# =========================================================================== #
# Feature colums
# =========================================================================== #
features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list('department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()

# with tf.Session() as sess:
#     sess.run((var_init, table_init))
#     print(sess.run(inputs))


# =========================================================================== #
# Training
# =========================================================================== #
# Define the data.
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
# Define the model.
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, _loss = sess.run((train, loss))
        if i % 10 == 0:
            print(_loss)

