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
    
To write a TensorFlow program based on pre-made Estimators, you must perform the following tasks:
* Create one or more input functions;
* Define the model's feature columns;
* instantiate an Estimator, specifying the feature columns and various hyperparameters;
* Call one or more methods on the Estimator object, passing the appropriate input function as the source of the data.

1. Create input functions  \
You must create input functions to supply data for training, evaluating, and prediction.    \
An input function is a function that returns a tf.data.Dataset object which outputs the following two-element tuple: 
* features - A Python dictionary in which:   
    * Each key is the name of a feature.
    * Each value is an array containing all of that feature's values. \
* label - An array containing the values of the label for every example.

2. Define the feature columns   \
A feature column is an object describing how the model should use raw input data from the features dictionary. When you build an Estimator model, you pass it a list of feature columns that describes each of the features you want the model to use. The tf.feature_column module provides many options for representing data to the model.

3. Instantiate an estimator 

4. Train, Evaluate, and Predict \
Now that we have an Estimator object, we can call methods to do the following:
* Train the model;
* Evaluate the trained model;
* Use the trained model to make predicitons.



