# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

tf.app.flags.DEFINE_string(
    'directory', '/tmp/data',
    'Directory to download data files and write the converted result.'
)

tf.app.flags.DEFINE_integer(
    'validation_size', 2000,
    'Number of examples to separate from the training data for the validation set.'
)

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, name):
    """Converts a dataset to tfrecords."""
    images = dataset.images
    labels = dataset.labels
    num_examples = dataset.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing ', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)})
        )
        writer.write(example.SerializeToString())
    writer.close()


def _parse_function(example_proto):
    features = {'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                'depth': tf.FixedLenFeature((), tf.int64, default_value=0),
                'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image_raw': tf.FixedLenFeature((), tf.string, default_value='')}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['height'], parsed_features['width'], \
           parsed_features['depth'], parsed_features['label'], parsed_features['image_raw']


def main(_):
    # Get the data.
    datasets = mnist.read_data_sets(FLAGS.directory,
                                    dtype=tf.uint8,
                                    reshape=False,
                                    validation_size=FLAGS.validation_size)
    # Convert to Examples and write the result to TFRecords.
    if not os.path.exists(os.path.join('/tmp/data/', 'train.tfrecords')):
        convert_to(datasets.train, 'train')
    if not os.path.exists(os.path.join('/tmp/data/', 'validation.tfrecords')):
        convert_to(datasets.validation, 'validation')
    if not os.path.exists(os.path.join('/tmp/data/', 'test.tfrecords')):
        convert_to(datasets.test, 'test')

    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.repeat()
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()

    # Initialize `iterator` with training data.
    validation_files = ['/tmp/data/validation.tfrecords']

    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={filenames: validation_files})
        while True:
            try:
                next_element = iterator.get_next()
                height, width, depth, label, image = next_element
                print(label, label.eval())
                sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    tf.app.run()
