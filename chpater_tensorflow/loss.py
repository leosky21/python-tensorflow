# coding: utf-8
# 设置损失函数 loss = (w+1)^2 ,令w初值是常数5. 反向传播就是求最优w,即使求最小loss对应的w值

import tensorflow as tf

# 定义待优化w权重
w = tf.Variable(tf.constant(5, dtype=tf.float32))

# 定义损失函数
loss = tf.square(w+1)

# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
# 生成会话,训练40轮
with tf.Session() as s:
    init_op = tf.global_variables_initializer()
    s.run(init_op)
    for i in range(40):
        s.run(train_step)
        w_ = s.run(w)
        loss_ = s.run(loss)
        print ("After %s steps: w is %f ,  loss is %f" % (i, w_, loss_))