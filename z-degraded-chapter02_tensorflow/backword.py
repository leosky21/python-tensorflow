# coding;utf-8
"""
反向传播就是训练网络, 优化网络参数
    - loss 正则化:
        y与y_的差距(loss_mse) = tf.reduce_mean(tf.square(y-y_))
    也可以是:
        cem = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y, labels=tf.argmax(y_, 1))
        y与y_的差距: tf.reduce_mean(cem)
    加入正则化:
        loss = y 与y_的差距 + tf.add_n(tf.get_collection("losses"))
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from chapter02_tensorflow import forward
from chapter02_tensorflow import generateds

STEPS = 10000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZED = 0.01


def backward():
    xs = tf.placeholder(tf.float32, shape=[None, 2])
    ys = tf.placeholder(tf.float32, shape=[None, 1])

    X, Y_, Y_C = generateds.generateds()
    y = forward.foward(x=xs, regularizer=REGULARIZED)
    global_step = tf.Variable(0, trainable=False)

    # 定义目标函数
    loss_mse = tf.reduce_mean(tf.square(y-ys))
    loss_total = loss_mse + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=300/BATCH_SIZE,
        decay_rate=LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_total)
    print("session out ===> train_step:",train_step)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={xs: X[start:end], ys: Y_[start:end] })
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={xs: X, ys: Y_})
                print("After %d steps, loss is %f" % (i, loss_v))
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={xx: grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_C))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()


if __name__ == '__main__':
    backward()