# [Conclusion](https://blog.csdn.net/Jerr__y/article/details/70809528)

* **name_scope:** 为了更好地管理变量的命名空间而提出的。比如在 tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
* """variable_scope:" 大大大部分情况下，跟 tf.get_variable() 配合使用，实现变量共享的功能。

## 1. Conclusion of `variables_example_0.1.py`

* `tf.placeholder():` 占位符。*trainable==False*
* `tf.Variable():` 一般变量用这种方式定义。 *可以选择 trainable 类型*
* `tf.get_variable():` 一般都是和 `tf.variable_scope()` 配合使用，从而实现变量共享的功能。 *可以选择 trainable 类型*

## 2. Conslusion of `variables_example_0.2.py`
`tf.name_scope()` and `tf.variable_scope()`
* `tf.name_scope()` 并不会对 `tf.get_variable()` 创建的变量有任何影响。`tf.name_scope()` 主要是用来管理命名空间的，这样子让我们的整个模型更加有条理。
* `tf.variable_scope()` 的作用是为了实现变量共享，它和 `tf.get_variable()` 来完成变量共享的功能。

## 3. Conclusion of `variables_example_0.3.py`
Use `tf.Variable()` to define variables without implementing variable sharing.

## 4. Conclusion of `variables_example_0.4.py`
Use `tf.get_variable()` to define variables and implement variable sharing perfectly! ! !


首先我们要确立一种 Graph 的思想。在 TensorFlow 中，我们定义一个变量，相当于往 Graph 中添加了一个节点。和普通的 python 函数不一样，在一般的函数中，我们对输入进行处理，然后返回一个结果，而函数里边定义的一些局部变量我们就不管了。但是在 TensorFlow 中，我们在函数里边创建了一个变量，就是往 Graph 中添加了一个节点。出了这个函数后，这个节点还是存在于 Graph 中的。
