import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value) ## load new_value into state

init_ = tf.global_variables_initializer() ## must have if define variable

with tf.Session() as sess:
    sess.run(init_)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))

