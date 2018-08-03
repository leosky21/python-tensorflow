# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist store data for training, checking, testing by numpy array
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

''' 
@PlaceHolder

create node of  @input_image and  @target_output 
    - shape: [None, 784]
        - None: num of batch , None means Variable number
    - 10 : 10 d one-hot vector to indicates the categories of pic
'''

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float")
x_image = tf.reshape(x, [-1, 28, 28, 1])
'''
initial weights, avoid dead neurons

@Variable

define the weights and biases of model .
    W: Image Dimensions Multiply 10 Dimensions , and output is 10
    b: 10 categories 
'''


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  ## TODO what's truncated_normal and stddev ??
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))




def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# we turn x into a 4d vector, with dimensions 2 and 3 corresponding to the width and height of the image
# the last dimension representing color channels of the image , 1 = grayscale image , 3 = RGB

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
'''
Softmax Regression:
    
cross_entropy:
'''
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''
Train model:
    It can use automatic differentiation to find the gradient value of the loss to each variable.

caculate the  gradient value, change of each variable, caculate new variables
'''

# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

'''
use `tf.argmax()` to evaluate our model
    - it can return the index of the max value
use `tf.equal()` to judge the equality 
'''
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
