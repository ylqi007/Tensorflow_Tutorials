import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# A normal way to define a conv layer
def conv_relu(kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    return weights, biases


def my_image_filter():
    # Define the convolution layer in the following way, very intuitive and layered
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu([5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu([5, 5, 32, 32], [32])


with tf.variable_scope("image_filters") as scope:
    # Below we call the my_image_filter function twice, but since the introduction of
    # the variable sharing mechanism, we can see that we just created the network structure.
    result1 = my_image_filter()
    scope.reuse_variables()
    result2 = my_image_filter()


# perfect for variable sharing! ! !
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)
