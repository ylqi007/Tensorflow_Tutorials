import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)   # ~/.keras/datasets/
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    print('In maybe_download\ttrain_path: ', train_path)    # /home/ylqi007/.keras/datasets/iris_training.csv
    print('In maybe_download\ttest_path: ', test_path)      # /home/ylqi007/.keras/datasets/iris_test.csv
    return train_path, test_path


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)   # <class 'pandas.core.frame.DataFrame'>
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)     # <class 'pandas.core.frame.DataFrame'>
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training."""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction."""
    features = dict(features)
    if labels is None:
        inputs = features   # No labels, use only features.
    else:
        inputs = (features, labels)

    # Convert the inputs to a Datasets.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "Batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
