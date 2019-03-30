from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.reset_default_graph()

# Create some variables:
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk,
# and do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk:
    saver.restore(sess, '/tmp/model.ckpt')
    print('### Model restored. ###')
    # Check the values of the variables:
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())


tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer=tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
    # Initialize v1 since the saver will not.
    v1.initializer.run()
    saver.restore(sess, "/tmp/model.ckpt")

    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())


# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]
