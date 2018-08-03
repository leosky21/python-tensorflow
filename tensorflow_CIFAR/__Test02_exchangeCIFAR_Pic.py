"""
    !!! Please read __Test_Read.py first !!!
    运行结束后,控制台出现:

    ERROR:tensorflow:Exception in QueueRunner: Enqueue operation was cancelled
       [[Node: input_producer/input_producer_EnqueueMany = QueueEnqueueManyV2[Tcomponents=[DT_STRING], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/device:CPU:0"](input_producer, input_producer/RandomShuffle)]]

    这意味着运行成功并且请在 `./read/ExchangeOutputDir/` 查看输出的图片

   - Aim : change dataset of CIFAR  into pics
   - Run : open Terminal and 'cd' to current dir and  'python3  __Test02_exchangeCIFAR_Pic'
   - version:
        - tensorFlow 1.8.0
        - python 3.6.5

    data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin 几个文件中每个文件有10000个样本.
        - 每个样本3073个字节:
            <1 x label><3072 x pixel>
        - 1字节: label
        - 3072字节: 图像数据

    <data set source: https://www.cs.toronto.edu/~kriz/cifar.html>
    The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin, as well as test_batch.bin. Each of these files is formatted as follows:
        <1 x label><3072 x pixel>
        ...
        <1 x label><3072 x pixel>
        In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.

        Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 30730000 bytes long.

        There is another file, called batches.meta.txt. This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names. It is merely a list of the 10 class names, one per row. The class name on row i corresponds to numeric label i.
"""

import tensorflow as tf
import os
import scipy.misc as sm


'''
    Steps:
        1. create a queue of pic list by tf.train.string_input_producer()
        2. read key and value by tf.FixedLengthRecordReader().read(queueName) from files 
            - because there are too many samples in a file , if a file corresponds to a sample we do it by tf.WholeFileReader().read(queueName)
        3. initial
        4. start_queue_runners
        5. Take out the date of each pic by f.write(sess.run(value)) 
'''

'''
    Returns: 
        An object representing a single example, with the following fields:
          height: number of rows in the result (32)
          width: number of columns in the result (32)
          depth: number of color channels in the result (3)
          key: a scalar string Tensor describing the filename & record number
            for this example.
          label: an int32 Tensor with the label in the range 0..9.
          uint8image: a [height, width, depth] uint8 Tensor with the image data
'''
def inputs_origin(filePath):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    fileNames = [os.path.join(filePath, 'data_batch_%d.bin' % _) for _ in range(1, 6)]  # in Python 3, rang() && xrange() == range()
    for f in fileNames:
        if not tf.gfile.Exists(f):
            raise ValueError('Error at __Test02_exchangeCIFAR_Pic.py : Failed to find file: ' + f)
    # create a queue of pic list by tf.train.string_input_producer()
    file_queue = tf.train.string_input_producer(fileNames)
    # read file and the size of step is record_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    # Change pic into float32
    reshape_image_data = tf.cast(result.uint8image, tf.float32)
    # the type of pic of returned is a Tensor
    # so whenever we sess.run(reshape_image) , we will take out a image
    return reshape_image_data

if __name__ == '__main__':
    with tf.Session() as sess:
        reshape_image = inputs_origin('tmp/cifar10_data/cifar-10-batches-bin')
        threads = tf.train.start_queue_runners()
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('read/ExchangeOutputDir/'):
            os.mkdir('read/ExchangeOutputDir/')
        for i in range(30):
            image_array = sess.run(reshape_image)
            sm.toimage(image_array).save('read/ExchangeOutputDir/%d.jpg' % i)
