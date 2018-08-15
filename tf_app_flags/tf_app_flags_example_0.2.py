"""
```shell
python3 tf_app_flags_example_0.2.py --color red
```

The usual way to write command-line programs in Python is with the Python standard library's argparse.

Interestingly, while the API of TensorFlow's tf.app.flags is very close to gflags,
it is itself implemented with argparse.
"""
# 2018-08-15 -- Qi

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--color',
                    default='green',
                    help='the color to make a flower')


def main():
    args = parser.parse_args(sys.argv[1:])
    print('a {} flower'.format(args.color))


if __name__ == '__main__':
    main()
