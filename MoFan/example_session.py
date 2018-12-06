import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)   # matrix multiple, similar to np.dot(m1, m2)

# Method 1
tf.logging.set_verbosity(tf.logging.DEBUG)
# with tf.Session() as sess:
#     print(sess.run(product))


# Method 2
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()