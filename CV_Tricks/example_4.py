"""
2019-03-28
https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

1. Meta graph: This is a protocal buffer which saves the complete TF graph with .meta extension.
2. Checkpoint file: This is a binary file which contains all the values of the weights, biases,
    gradients and all the other variables saved. This file has an extension .ckpt. Instead of
    single .ckpt file, it has two files:
        * mymodel.data-00000-of-000001
        * mymodel.index
    TF also has a file named checkpoint which simply keeps a record of latest checkpoint files saved.
3. TF variables are only alive inside a session. So, you have to save the model inside a session
    by calling save method on saver object you just created: saver.save(sess, 'my-test-model').
4. Importing a pre-trained model:
    * Create the network: `saver = tf.train.import_meta_graph('my_test_model-1000.meta')`
    * Load the parameters:
5. Working with restored models:

"""
import tensorflow as tf

w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, './my_test_model/my-test-model')


###############################################################################
# Example 1: Save
###############################################################################
# Prepare to feed input, i.e. feed_dict and placeholders
w11 = tf.placeholder("float", name='w11')
w12 = tf.placeholder("float", name='w12')
b1 = tf.Variable(2.0, name='bias')
_feed_dict = {w11: 4, w12: 8}

# Define a test operation that will restore
w13 = tf.add(w11, w12)
w14 = tf.multiply(w13, b1, name='op_to_restore')

# Create a saver object which will save all the variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(w14, feed_dict=_feed_dict))

    # Now, save the graph
    saver.save(sess, './my_test_model/demo1', global_step=1000)


###############################################################################
# Example 2: Restore
###############################################################################
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./my_test_model/demo1-1000.meta')
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./my_test_model/'))

# Access and create placeholders variables and create feed-dict to feed new data
graph = tf.get_default_graph()
w11 = graph.get_tensor_by_name("w11:0")
w12 = graph.get_tensor_by_name("w12:0")
feed_dict = {w11: 13.0, w12: 17.0}

op_to_restore = graph.get_tensor_by_name(name='op_to_restore:0')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(op_to_restore, feed_dict=feed_dict))


###############################################################################
# Example 3: Add more operations to the graph
###############################################################################
# Load meta graph and restore weights.
saver = tf.train.import_meta_graph('./my_test_model/demo1-1000.meta')
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./my_test_model/'))

# access and create placeholders variables
graph = tf.get_default_graph()
w11 = graph.get_tensor_by_name("w11:0")
w12 = graph.get_tensor_by_name("w12:0")
feed_dict = {w11: 13.0, w12: 17.0}

# access the op that you want to run
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# Add more to the current graph
add_on_op = tf.multiply(op_to_restore, 2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(add_on_op, feed_dict=feed_dict))


###############################################################################
# Example 4: Restore part of the old model and add-on for fine-tuning
###############################################################################
