import tensorflow as tf

# Allow GPU grow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print("########## Placeholder ##########")
# 1.placeholder
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4])
print(v1.name)      # Placeholder:0
print('Placeholder without setting name', v1.name)

v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
print(v1.name)      # ph:0

v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
print(v1.name)      # ph_1:0
print("type of tf.placeholder()", type(v1))
print(v1)


print("########## tf.Variable() ##########")
# 2. tf.Variable()
v2 = tf.Variable([1, 2], dtype=tf.float32)
print(v2.name)      # Variable:0
print('tf.Variable() without setting name', v2.name)
v2 = tf.Variable([1, 2], dtype=tf.float32, name='V')
print(v2.name)      # V:0
v2 = tf.Variable([1, 2], dtype=tf.float32, name='V')
print(v2.name)      # V_1:0
print("type of tf.Varialbe()", type(v2))
print(v2)


print("########## tf.get_variable() ##########")
# 3.tf.get_variable(), the name is must provided
v3 = tf.get_variable(name='gv', shape=[])
print(v3.name)      # gv:0
# v4 = tf.get_variable(name='gv', shape=[2])
# print(v4.name)
print("type of tf.get_variable()", type(v3))     # <class 'tensorflow.python.ops.variables.Variable'>
print(v3)


print("########## conclusion ##########")
vs = tf.trainable_variables()
print(len(vs))
for v in vs:
    print(v, v.name)
