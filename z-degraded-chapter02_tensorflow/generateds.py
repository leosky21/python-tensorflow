# coding:utf-8
"""
    生成数据集,标记数据集
"""
import numpy as np
import matplotlib.pyplot as plt

seed = 3


def generateds():
    # 1. 基于seed 产生随机数
    rdm = np.random.RandomState(seed=seed)
    # 2. 随机数返回一个 xxx行xx列的矩阵,用作输入数据集
    X = rdm.rand(300, 2)
    # 3. 从输入数据集中,取出一行, 作为Y的正向值
    Y_ = [int((x0*x0 + x1*x1) < 2) for (x0, x1) in X]
    # 4. 遍历Y中的每一个元素, 1 则赋值为 'red', 其余的赋值为'blue'
    Y_C = [['red' if y else 'blue'] for y in Y_]
    # 5. 数据集X和标签Y进行形状整理, 第一个元素为-1, 表示跟随第二列计算, 第二个元素表示多少列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)
    # 6. 画出第一行,第二行的元素.
    # plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_C))
    # plt.show()
    return X, Y_, Y_C

