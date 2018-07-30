# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
 generally , we use linear function instead of custom function
     define four parameters in function add_layer:
        -   inputs: the value to be input
        -   size: the size of the value
        -   out_size: the size of output value
        -   activation_function: the function of activation, we set it to use linear equations by default 
'''


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope("layer"):
        with tf.name_scope('weights'):
    ## normal_distribution is better than zeros ##
            Layer_Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+"/weights", Layer_Weights)
    ## donnot recommend zero ##
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')
            tf.summary.histogram(layer_name + "/biases", biases)
    ## the unactivatived value of the neural netework ##
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Layer_Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs

## train data, to simulate the data we need to train
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # 300line
noise = np.random.normal(0, 0.05, x_data.shape) # variance(mean) is 0.05 , type is the same to x_data
## the norm data
y_data = np.square(x_data) - 0.5+noise

'''
 the data trim to be input , tell the system : i am going to enter some data like this
    -  'tf.float32': the type of data we are going to input
    -   'None': it means whatever the number of input data is
    -   '1' : the number of features, here we only one
'''
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

'''
 input value : xs
    -  10 neurals
    -  use relu as activation_function
'''
layer1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, n_layer=2, activation_function=None)

## the difference between prediction  and y_data  which is the norm(real) data ##
with tf.name_scope(name='loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope(name='train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  ## learning rate. to minimize(reduce) the loss

with tf.name_scope(name='init_op'):
    init_op = tf.global_variables_initializer()


sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)
## now , start
sess.run(init_op)
## create a image framework
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
## to see the improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})

        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        writer.add_summary(result, i)
        plt.pause(0.4)
        try:
            ax.lines.remove(lines[0])
        except Exception :
            pass
sess.close()

'''
    SGD 
        W += -LEARNING_RATE * dx  ---Momentum  ----> m = b1 * m -LEARNING_RATE * dx , W += m
    AdaGrad
        v += dx ^2  , W += -LearningRATE * dx/ 
'''