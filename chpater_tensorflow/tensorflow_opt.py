# coding:utf-8
# 预测多的影响和少的影响一样
# 0. 导入模块, 生成模拟数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 234599

# 基于seed 产生随机数
rng = np.random.RandomState(seed)
# 随机数返回32行2列的矩阵, 表示32组 提及和重量 ,所谓输入数据集
X_data = rng.rand(32, 2)
# 从X这个32行2列的矩阵中, 取中一行,判断如果和小于1 ,给Y赋值1 ; 如果和不小于1 给y赋值0
# 构建作为输入数据集的标签(正确答案)
Y_data = [[x0 + x1 + (rng.rand()/10.0 - 0.05)] for (x0, x1) in X_data]
print("X:\n",  X_data)
print("Y_:\n", Y_data)

# 1. 定义 神经网络的输入, 参数,和输出, 定义前向传播过程

x_input = tf.placeholder(tf.float32, shape=(None, 2))
y_input = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

layer1 = tf.matmul(x_input, w1)

# 2. 定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(layer1 - y_input))  # 损失函数均方误差
train_step = tf.train.GradientDescentOptimizer(0.002).minimize(loss) # 梯度下降
#train_step = tf.train.MomentumOptimizer(0.001, 1).minimize(loss)
#train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

# 生成会话,训练STEPS 轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前未经训练的参数取值
    print("w1: ", sess.run(w1))
    # print("w2: \n", sess.run(w2))
    # 训练模型
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x_input: X_data[start: end], y_input: Y_data[start:end]})
        if i % 5000 == 0:
            total_loss = sess.run(loss, feed_dict={x_input: X_data, y_input: Y_data})
            print("After %d training step(s), loss on all data is %g" % (i, total_loss))
    # 输出训练后的参数取值
        print("\n")
        print("w1: %s \n" % sess.run(w1))
    # print("w2: %s \n" % sess.run(w2))


