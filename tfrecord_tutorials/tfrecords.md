# [TFRecord files Tutorial](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)

---
## 1. tf.train.Example
> 1. `tf.train.Example` stores features in **a single attribute** `features` of type `tf.train.Features`;
> 2. `tf.train.Features` is a collection of named features. It has **a single attribute** `feature` that expects a dictionary where the *key* is the name of the features and the *value* a `tf.train.Feature`;
> 3. `tf.train.Feature` wraps a list of data of specific type so Tensorflow can understand it. It has a single attribute, which is a union of *bytes_list/float_list/int64_list*;
> 4. `tf.train.BytesList, tf.train.FloatList, tf.train.Int64List` are at the core of a `tf.trian.Feature`. All three have a single attribute `value`, which expects a list of respective *bytes, float, and int*.

---
## 2. tf.train.SequenceExample