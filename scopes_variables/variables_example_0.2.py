"""
`tf.name_scope()` has no effect on `tf.get_variable()`. It just design for mange namespace.

`tf.variable_scope()` is designed for variable-sharing with `tf.get_variable()`.
"""
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


with tf.name_scope('nsc1'):
    v1 = tf.Variable([1], name='v1')
    v2 = tf.get_variable(name='v2', shape=[])
    with tf.variable_scope('vsc1'):
        v3 = tf.Variable([1], name='v3')
        v4 = tf.get_variable(name='v4', shape=[])
print('v1.name: ', v1.name)     # nsc1/v1:0         tf.Variable()
print('v2.name: ', v2.name)     # nsc1/vsc1/v2:0    tf.get_variable()
print('v3.name: ', v3.name)     # vsc1/v3:0         tf.Variable()
print('v4.name: ', v4.name)     # v4:0              tf.get_variable()


with tf.name_scope('nsc1'):
    v2 = tf.Variable([1], name='v2')
print('v2.name: ', v2.name)
