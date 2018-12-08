"""
https://www.tensorflow.org/guide/tensors
"""

import tensorflow as tf

# Rank 0
mammal = tf.Variable('Elephant', tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.1415926, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

with tf.Session() as sess:
    tf.logging.set_verbosity(tf.logging.ERROR)
    sess.run(tf.global_variables_initializer())
    print(mammal, '\t', sess.run(mammal))           # <tf.Variable 'Variable:0' shape=() dtype=string_ref> 	 b'Elephant'
    print(ignition, '\t', sess.run(ignition))
    print(floating, '\t', sess.run(floating))
    print(its_complicated, '\t', sess.run(its_complicated))


# Rank 1
mystr = tf.Variable(["Hello"], tf.string)
# For cool_numbers, the rank is 1, and the size of this dimension is 2
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(mystr, '\t', sess.run(mystr))        # <tf.Variable 'Variable_4:0' shape=(1,) dtype=string_ref> 	 [b'Hello']
    print(cool_numbers, '\t', sess.run(cool_numbers))
    print(first_primes, '\t', sess.run(first_primes))
    print(its_very_complicated, '\t', sess.run(its_very_complicated))


# Higher ranks
mymat = tf.Variable([[7],[11]], tf.int16)       # shape=(2, 1)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)     # shape=(2, 2)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)  # shape=(4, 1)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)     # Tensor("Rank:0", shape=(), dtype=int32) 	 2
mymatC = tf.Variable([[7],[11]], tf.int32)
my_image = tf.zeros([2, 5, 5, 3])  # batch x height x width x color

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(mymat, '\t', sess.run(mymat))     # <tf.Variable 'Variable_8:0' shape=(2, 1) dtype=int32_ref>     [[7] [11]]
    print(myxor, '\t', sess.run(myxor))     # <tf.Variable 'Variable_9:0' shape=(2, 2) dtype=bool_ref> 	 [[False True] [True False]]
    print(linear_squares, '\t', sess.run(linear_squares))
    print(squarish_squares, '\t', sess.run(squarish_squares))
    print(rank_of_squares, '\t', sess.run(rank_of_squares))     # Tensor("Rank:0", shape=(), dtype=int32) 	 2
    print(mymatC, '\t', mymatC.eval())      # <tf.Variable 'Variable_12:0' shape=(2, 1) dtype=int32_ref> 	 [[7] [11]]
    # print(my_image, '\t', my_image.eval())   # shape=(2, 5, 5, 3)

