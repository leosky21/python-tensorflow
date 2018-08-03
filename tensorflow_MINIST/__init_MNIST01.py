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
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_input: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    print("实际样本值 ::", sess.run(tf.argmax(v_ys, 1)))
    print("预测的样本值 ::", sess.run(tf.argmax(y_pre, 1)))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x_input: v_xs, y: v_ys})
    return result


# number 1 to 10 data
'''
    one-hot : One-bit efficient coding
    origin  |   one-hot
       0        (1,0,0,0,0,0,0,0,0,0)
       1        (0,1,0,0,0,0,0,0,0,0)
       2        (0,0,1,0,0,0,0,0,0,0)
       3        (0,0,0,1,0,0,0,0,0,0)
      ...               ......
       8        (0,0,0,0,0,0,0,0,1,0)
       9        (0,0,0,0,0,0,0,0,0,1)
'''
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
    # define placeholder for iputs to network
        - 784 : 28*28 = 784px per pic
        - each pic indicats a number range in [0,9], sum is 10 ##
'''
x_input = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

'''
    add a output layer
'''
prediction = add_layer(x_input, 784, 10, activation_function=tf.nn.relu)

'''
    the error between prediction and real data
    cross_entropy : loss
'''
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

#sess = tf.Session()
sess = tf.InteractiveSession()
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs_data, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x_input: batch_xs_data, y: batch_ys})
    if i % 100 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))


'''
    - matmul():
        a: Tensor of type float16, float32, float64, int32, complex64, complex128 and rank > 1.
        b: Tensor with same type and rank as a.
        transpose_a: If True, a is transposed before multiplication.
        transpose_b: If True, b is transposed before multiplication.
        adjoint_a: If True, a is conjugated and transposed before multiplication.
        adjoint_b: If True, b is conjugated and transposed before multiplication.
        a_is_sparse: If True, a is treated as a sparse matrix.
        b_is_sparse: If True, b is treated as a sparse matrix.
        name: Name for the operation (optional).
    
    矩阵中的共轭转置(conjugated and transposed):
        把矩阵转置后，再把每一个数换成它的共轭复数(所谓“轭”，指的是古代牛车上放在并行的牛脖颈上的曲木。共轭关系，通俗来说一般用以描述两件事物以一定规律相互配对或孪生（一般共轭对整体很相似，但在某些特征上却性质相反）。)

'''