#  -*- coding: utf-8 -*-
"""
    this file is to save data of mnist as a image file

    TODO :
       @error AttributeError: module 'scipy.misc' has no attribute 'toimage'
       @solution  pip(3) install pillow

"""
import scipy.misc as sm
from tensorflow.examples.tutorials.mnist import input_data
import os
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 将原始图片保存在MNIST_data/raw 文件夹,若没有自动创建
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

# 保存前20张图片
# 请注意:  mnist.train.images[i, :]就表示第i张图片 ,从0开始
for i in range(20):
    image_array = mnist.train.images[i, :]
    # Tensorflow 中将图像拉成了784维的向量, 重新还原成28 x 28 维的图像
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式  mnist_train_i.jpg
    filename = save_dir+'mnist_trian%d.jpg' % i
    # 将image保存为图片
    # 先用scipy转换成图像, 再调用save直接保存

    image = sm.toimage(arr=image_array, cmin=0.0, cmax=1.0).save(filename)
