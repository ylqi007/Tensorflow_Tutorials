`iris_data.py`
* `def maybe_download()`    \
    Download file from specific URL to a local directory. And return the local directory to the download file.
* `def load_data(y_name)`   \
    Read data from `.csv` file and return `DataFrame`.
* `def train_input_fn()`    \
    Convert `features` into a dataset and batch the `features`.
* `def eval_input_fn()`     \
    Similar to `train_input_fn()` and returns a dataset.

`premade_estimator.py`
* `parser=argparse.ArgumentParser()`    \
    Read parameters from command line.
* `def main(argv)`
    The whole process of the network.
* Fetch the data
* Feature columns describe how to use the input.    \
    `my_feature_columns` is a `list`, and the elements in the `my_feature_columns` is type `NumericColumn`.
* `classifier = tf.estimator.DNNClassifier(...)`    \
    Define the classifier, given the `features_columns`, `model_dir`, `hidden_units` and `n_classes`.
    * `feature_columns`, the input data to classifier;
    * `model_dir`, the directory where to save the data;
    * `hidden_units`, the number of units in each hidden layer.
    * `n_classes`, the size of the output, i.e. the number of classes for the classifier.
* `classifier.train(...)`   \
    Return a `Classifer` itself, for chaining.
* `classifier.evaluate(...)`    \
    Evaluate on the test dataset.
* `predictions=classifer.predict(...)`  \
    Output the predictions given the input data. Like `logits`, `probabilities` and `class_id`.
    
    