# coding:utf-8


"""
    @introduce:
        prepare data from MINIST  which is a database of writing fonts
            - each pic is 28 * 28 = 784px = x_input
            - total number : 55000
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    @layer
        each layer all have @layer_weights, @biases,  
'''


def add_layer(in_data, in_size, out_size, activation_function=None):
    layer_weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([10]))
    Wx_plus_b = tf.nn.softmax(tf.matmul(in_data, layer_weights) + biases)
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


'''
    @compute: 
         Organization final output
    
'''


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_input: v_xs, keep_prob: 0.7})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x_input: v_xs, y: v_ys, keep_prob: 0.7})
    return result


'''
    @weight_variable:
        initial variable
'''


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


'''
    @TODO: 
        check tf.constant() ,tf.Variable() and some other... Varivales
    @bias:
        initial bias
'''


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
    
'''


def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # Must have stride[0] = stride[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# stride = 2*2, twice
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# number 1 to 10 data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
    # define placeholder for iputs to network
        - 784 : 28*28 = 784px per pic
        - each pic indicats a number range in [0,9], sum is 10 ##
'''
keep_prob = tf.placeholder(tf.float32)
x_input = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x_input, [-1, 28, 28, 1])

'''
    conv1 layer: 
        patch : 5*5
        channel :   Black and white photos == 1
                    (RGB = 3)
        featureMap :  32 
        padding = 'SAME' : don't change length,width of pic , but height
        
'''

W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1 , out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # out size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)  # out size 14*14*32

'''
    conn2 layer :
        patch : 5 * 5
        
'''

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 1 , out size 32
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # out size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # out size 7*7*64

'''
    fully connected layer
    reshape: reshape 3D pic into  7*7*64
'''

## func1 layer ##
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # [n_sample ,7,7,64 ]  >>>  [n_sample,7*7*64] ==== 降维
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # out size 14*14*64
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''
    the error between prediction and real data
    cross_entropy : loss
'''
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# sess = tf.Session()
sess = tf.InteractiveSession()
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs_data, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_input: batch_xs_data, y: batch_ys, keep_prob: 0.7})
    if i % 100 == 0:
        print(compute_accuracy(
            mnist.test.images[:10000], mnist.test.labels[:10000]))
