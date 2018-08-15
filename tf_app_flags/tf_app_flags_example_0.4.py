"""
Rather than access sys.argv directly, it's nice to use something that will automatically process it.

The tf.app.run() functionality does a minimal transformation in this case, passing sys.argv as an argument to main().

The choice of main() as the function to run is by convention,
and can be overridden, for example like tf.app.run(main=my_cool_function).
"""
# 2018-08-15 -- Qi

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string(name='color',
                    default='green',
                    help='the color to make a flower')


def main(args):
    print('a {} flower'.format(flags.FLAGS.color))


if __name__ == '__main__':
    tf.app.run()
