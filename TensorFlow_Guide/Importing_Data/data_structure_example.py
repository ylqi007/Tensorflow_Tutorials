"""Examples from Importing Data(https://www.tensorflow.org/guide/datasets)"""

import tensorflow as tf

print('##### Dataset1:')
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"
print(dataset1)

print('##### Dataset2:')
dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

print('##### Dataset3:')
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

print('##### Dataset4:')
dataset4 = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset4.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset4.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

