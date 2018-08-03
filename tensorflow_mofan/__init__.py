import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

baises = tf.Variable(tf.zeros([1]))

y = Weights * x_data + baises

loss = tf.reduce_mean(tf.square(y- y_data)) ## caculate the dis
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5 is the learning_rate
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()  # to init all the variable int the pic with struct we made what above

### create tensorflow end ###

sess = tf.Session()

sess.run(init)  ## to do  ##

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(baises))