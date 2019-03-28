# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
2019-03-28

https://github.com/sankit1/cv-tricks.com/blob/master/Tensorflow-tutorials/Tensorflow-slim-run-prediction/nets/alexnet.py

Contains a model definition for AlexNet.

Usage:
    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        outputs, end_points = alexnet.alexnet_v2(inputs)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


slim = tf.contrib.slim
# trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def trunc_normal(stddev):
    return tf.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initilizer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
    """
    AlexNet version 2.

    :param inputs: a tensor of size [batch_size, height, width, channels].
    :param num_classes: number of predicted classes.
    :param is_training: whether or not the model is being trained.
    :param dropout_keep_prob: the probability that activations are kept in the
        dropout layers during training.
    :param spatial_squeeze: whether or not should squeeze the spatial dimensions
        of the output. Useful to remove unnecessary dimensions for classification.
    :param scope: Optional scope for the variables.
    :return:
        The last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            # 1
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(num_classes, [3, 3], 2, scope='pool1')
            # 2
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            # 3
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            # 4
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            # 5
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                # 6
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
                net = slim.conv2d(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                # 7
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                # 8
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net

            return net, end_points


alexnet_v2.default_image_size = 224
