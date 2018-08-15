# [TFRecord files Tutorial](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)

---
## 1. tf.train.Example
1. `tf.train.Example` stores features in **a single attribute** `features` of type `tf.train.Features`;
2. `tf.train.Features` is a collection of named features. It has **a single attribute** `feature` that expects a dictionary where the *key* is the name of the features and the *value* a `tf.train.Feature`;
3. `tf.train.Feature` wraps a list of data of specific type so Tensorflow can understand it. It has a single attribute, which is a union of *bytes_list/float_list/int64_list*;
4. `tf.train.BytesList, tf.train.FloatList, tf.train.Int64List` are at the core of a `tf.trian.Feature`. All three have a single attribute `value`, which expects a list of respective *bytes, float, and int*.

---
## 2. tf.train.SequenceExample
`tf.train.SequenceExample` is also one of the main components for structuring a TFRecord, but it has **two attributes**, and there is only **a single attribute** for `tf.train.Example`.
1. **Two attributes** of `tf.train.SequenceExample`:
	* `context`: This attribute expects type `tf.train.Features`;
	* `feature_lists`: The type of this attribute is `tf.train.FeatureLists`.
2. The data stored in `context` is as `tf.train.Feature`, just like `tf.train.Example`;
3. As for `feature_lists`, this component has **a single attribute** `feature_list` that expect a dict. And `tf.train.FeatureLists` is a collection of named instances of `tf.train.FeatureList`.
4. `tf.train.FeatureList` has **a single attribute** `feature` that expects a list with entries of type `tf.train.Feature`.