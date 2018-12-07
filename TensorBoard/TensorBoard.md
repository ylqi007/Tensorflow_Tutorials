> TensorBoard operates by reading TensorFlow events files, which contain summary data that you can generate when running TensorFlow.

1. First, create the TensorFlow graph that you'd like to collect summary data from, and decide which nodes you would like to annotate with (summary operations)[https://github.com/tensorflow/docs/tree/master/site/en/api_guides/python].

Operations in TensorFlow don't do anything until you run them, or an op that depends on their output. The summary nodes that we've just created are peripheral to the graph: none of the ops you are currently running depend on them.

2. Then, you can just run the merged summary op, which will generate a serialized `Summary` protobuf object with all of your summary data at a given step.

3. Finally, to write this summary data to disk, pass the summary protobuf to a `tf.summary.FileWriter`.

The `FileWriter` takes a `logdir` in its constructor - this logdir is quite important, it's the directory where all of the events will be written out. Also `FileWriter` can optionally take a `Graph` in its constructor.
