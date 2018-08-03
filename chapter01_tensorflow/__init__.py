#!/usr/bin/local/python
# -*- coding:utf-8 -*-
import tensorflow as tf
# # 张量 表示数据: 多维数组, 列表, 矩阵
# #  0-D: 标量  scalar      s=1,2,3
# #  1-D: 向量  vector      v=[1,2,3]
# #  2-D: 矩阵  matrix      m = [[1,2,3] , [4,5,6],[7,8,9]]
# #  n-D: 张量  tensor      t= [[[[[[.....

# 数据类型: tf.float32 , tf.int32

a = tf.constant([[1.0, 2.0]])
b = tf.constant([[2.0], [3.0]])

result = tf.matmul(a, b)
print result

with tf.Session() as session:
    print session.run(result)
    '''
    变量初始化
    '''
    # int_op = tf.global_variables_initializer()
    # session.run(int_op)

# 参数: 即 权重W , 用变量白表示随机给初值

#                正态分布    产生2x3的矩阵  标准差2    均值0               随机种子1
w = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, dtype=float, seed=1) )
'''
            tf.truncated_normal() : 去掉过大偏离点的正态分布
            tf.random_uniform() : 平均分布
            tf.zeros([3, 2], int32) 生成全0数组
            tf.ones([3,2], int32) 生成全1数组
            tf.fill([3,2], 6) 生成固定填充值
            tf.constant([3,2,1]) 直接给值
'''

# 神经网络的实现过程
"""
1. 准备数据集, 提取特征, 作为输入喂给神经网络
2. 搭建NN结构,从输入到输出(搭建计算图,再用会话执行)
    ( NN 前向传播算法  ---> 计算输出)
3. 大量特征喂给NN, 迭代优化NN参数
    ( NN 反向传播算法 --->  优化从参数训练模型)
4. 训练好的模型预测和分类

"""
