# coding: utf-8
"""
    Aim of this file: to download dataset of cifar10
    Run : open Terminal and 'cd' to current dir and  'python3  __init_CIFAR_downlaod'
    version:
        - TensorFlow 1.8.0
        - python 3.6.5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib
import os.path
import os
import sys
import tarfile

import tensorflow as tf

# tf.app.flags.FLAGS 是一个Tensorflow内部的全局变量存储器, 同时可以用于命令行参数的处理
# 更换路径
# 定义全局变量
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# 基本模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir  # /tmp/cifar10_data
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # 从URL中获得文件名
    filename = DATA_URL.split('/')[-1]
    # 合并文件路径
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 定义下载过程中打印日志的回调函数
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        # 下载数据集
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                                 reporthook=_progress)
        print()
        # 获得文件信息
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        # 解压缩
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


maybe_download_and_extract()
