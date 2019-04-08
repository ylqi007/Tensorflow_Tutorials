"""
2019-04-08
"""
import tensorflow as tf


# Metadata describing the text columns
COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'label']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields.
    fileds = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary.
    features = dict(zip(COLUMNS, fileds))

    # Separate the label from the features.
    label = features.pop('label')

    return features, label





