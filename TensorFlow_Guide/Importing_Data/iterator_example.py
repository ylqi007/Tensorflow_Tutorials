import tensorflow as tf

# Example 1
# dataset = tf.data.Dataset.range(100)
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(100):
#         value = sess.run(next_element)
#         assert i == value
#         print(value)


# Example 2
# max_value = tf.placeholder(tf.int64, shape=[])
# dataset = tf.data.Dataset.range(max_value)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# # Initialize an iterator over a dataset with 10 elements
# with tf.Session() as sess:
#     sess.run(iterator.initializer, feed_dict={max_value: 10})
#     for i in range(10):
#         value = sess.run(next_element)
#         assert i == value
#         print(value)
#
# # Initialize the same iterator over a dataset with 100 elements.
# with tf.Session() as sess:
#     sess.run(iterator.initializer, feed_dict={max_value: 100})
#     for i in range(100):
#         value = sess.run(next_element)
#         assert i == value
#         print(value)


# Example 3
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()
training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
with tf.Session() as sess:
    for _ in range(1):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for i in range(100):
            value = next_element
            print('%d: %d' % (i, value.eval()))
            print(next_element)
            # sess.run(next_element)

        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
            value = next_element.eval()
            print(value)
            # sess.run(next_element)


