"""
2019-03-28
https://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow/tensorflow-tutorial/

1. Data in TF is represented by n-dimensional arrays called Tensors.
2. Graph is made of data (i.e. Tensors) and mathematical operations.
    * Nodes in the graph: represent mathematical operations;
    * Edges in the graph: represent the Tensors that flow between operations.
3. In TF, you first need to create a blueprint of whatever you want to create.
While you are creating the graph, variables don't have any value. Later when
you have created the complete graph, you have to run it inside a session, only
then the variables have any values.
4. A graph is used to define operations, but the operations are only run within
a session. Graphs and sessions are created independently of each other.
5. You can't print/access constant a variable unless you run it inside a session.
6. Variables need to be separately initialized by an init op.
7. Placeholders are tensors which are waiting to be initialized/fed.
8. TF has very strong in-built capabilities to run code on a gpu or a cpu or a cluster of gpu.
"""

import tensorflow as tf


def test_graph():
    graph = tf.get_default_graph()
    print(graph)
    print(graph.get_operations())
    for op in graph.get_operations():
        print(op.name)


def test_tensors():
    # Constants
    a = tf.constant(1.0)
    # Variables
    b = tf.Variable(2.0, name="test_var")
    # Multiply
    y = tf.multiply(a, b)
    # Placeholder
    a1 = tf.placeholder(tf.float32)
    b1 = tf.placeholder(tf.float32)
    y1 = tf.multiply(a1, b1)
    fed_dict = {a1: 2, b1: 3}
    # init_op
    init_op = tf.global_variables_initializer()

    for op in tf.get_default_graph().get_operations():
        print("\t#####: ", op.name)
    print("#####\t#####\t#####\t#####")
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(a))
        print(sess.run(b))
        print(sess.run(y))
        print("##### test placeholder #####")
        print(sess.run(y1, feed_dict=fed_dict))


if __name__ == '__main__':
    print("########## test_graph() ##########")
    test_graph()
    print("########## test_tensors() ##########")
    test_tensors()

