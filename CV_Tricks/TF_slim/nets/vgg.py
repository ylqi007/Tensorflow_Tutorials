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

https://github.com/sankit1/cv-tricks.com/blob/master/Tensorflow-tutorials/Tensorflow-slim-run-prediction/nets/vgg.py

Contains model definitions for versions of the Oxford VGG network.

Usage:
    with slim.arg_scope(vgg.vgg_arg_scope()):
        outputs, end_points = vgg.vgg_a(inputs)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        outputs, end_points = vgg.vgg_b(inputs)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
    """
    Define the VGG arg scope.

    :param weight_decay: The l2 regularization coefficient.
    :return: An arg_scope
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularization=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_a():
    pass


vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    """
    Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    :param inputs: A tensor of size [batch_size, height, width, channels]
    :param num_classes: number of predicted classes.
    :param is_training: whether or not the model is being trained.
    :param dropout_keep_prob: the probability that activations are kept in the
        dropout layers during training.
    :param spatial_squeeze: whether or not should squeeze the spatial dimensions
        of the outputs. Useful to remove unnecessary dimensions for classification.
    :param scope: Optional scope for the variables.
    :return:
        the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            # Block #1
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')    # [224, 224]
            net = slim.max_pool2d(net, [2, 2], scope='pool1')       # [112, 112]
            # Block #2
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')      # [112, 112]
            net = slim.max_pool2d(net, [2, 2], scope='pool2')       # [56, 56]
            # Block #3
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')      # [56, 56]
            net = slim.max_pool2d(net, [2, 2], scope='pool3')       # [28, 28]
            # Block #4
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')      # [28, 28]
            net = slim.max_pool2d(net, [2, 2], scope='pool4')       # [14, 14]
            # Block #5
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')      # [14, 14]
            net = slim.max_pool2d(net, [2, 2], scope='pool5')       # [7, 7]
            # Use conv2d instead of fully connected layers.
            # Block #6
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            # Block #7
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            # Block #8
            net = slim.conv2d(net, num_classes, [1, 1],
                              activation_fn=None,
                              normalizer_fn=None,
                              scope='fc8')
            # Convert end_points_collection into a end_point dict
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net

            return net, end_points


vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19'):
    """
    Oxford Net VGG 19-Layers version E Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:

    :param inputs: A tensor of size [batch_size, height, width, channels]
    :param num_classes: number of predicted classes.
    :param is_training: whether or not the model is being trained.
    :param dropout_keep_prob: the probability that activations are kept in the
        dropout layers during training.
    :param spatial_squeeze: whether or not should squeeze the spatial dimensions
        of the outputs. Useful to remove unnecessary dimensions for classification.
    :param scope: Optional scope for the variables.
    :return:
        the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            # Block #1
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # Block #2
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # Block #3
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # Block #4
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # Block #5
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            # Block #6
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            # Block #7
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            # Block #8
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


vgg_19.default_image_size = 224


# Alias
vgg_d = vgg_16
vgg_e = vgg_19
