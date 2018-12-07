import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)

print(state.name)
print(one)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()    # must have if define varialbes

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))