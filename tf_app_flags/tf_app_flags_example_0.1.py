"""
```shell
python3 tf_app_flags_example_0.1.py --color red
```

This direct approach is concise, for this simple example, but brittle
(what if there are multiple arguments, in different order?)
and opaque (what was argument 7 again?).

"""
# 2018-08-15 -- Qi
import sys


def main():
    assert sys.argv[1] == '--color'
    print('a {} flower'.format(sys.argv[2]))


if __name__ == '__main__':
    main()

