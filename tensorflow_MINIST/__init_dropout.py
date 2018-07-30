# coding:utf-8
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


## define some methods

def add_layer(inputs, in_size,  out_size, layer_name, activation_function=None):
    ## normal_distribution is better than zeros ##

    Layer_Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    ## donnot recommend zero ##

    biases = tf.Variable(tf.zeros([1.]))

    ## the unactivatived value of the neural netework ##
    Wx_plus_b = tf.matmul(inputs, Layer_Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


## load data ##
digits = load_digits()
X = digits.data
# change data  with 01 . e.g. number 2 <----> 00100 00000
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

## define placeholder for inouts to network
'''
    keep_prob : the number which u want to leave , if it is 0.6, it will drop 0.4 and save 0.6
    xs  : the value u want to put in
    ys  : the value u want to put in  
'''
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  ## 8*8
ys = tf.placeholder(tf.float32, [None, 10])

## add a leayer
l1 = add_layer(xs, 64, 50, "l1", activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

## the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

## summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

sess.run(tf.global_variables_initializer())

for _ in range(1000):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.3})
    if _ % 50 == 0:
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.3})
        train_test = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 0.3})
        train_writer.add_summary(train_result, _)
        test_writer.add_summary(train_test, _)
