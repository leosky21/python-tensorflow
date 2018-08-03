# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## import MNIST data

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./autoEncoder/data/", one_hot=True)

# Visualize decoder setting ##
# Parameters ##

learning_rate = 0.01
train_epochs = 5  # 5组训练
batch_size = 256
display_step = 1
example_to_show = 10

# Network Parameters##

n_input = 784  # MNIST data input (img shape: 28*28) ##

# tf Graph input(only pictures)  #
X = tf.placeholder("float", [None, n_input])

# hidden layer setting

n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}


# Building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)     # 128 Features
decoder_op = decoder(encoder_op)    # 784 Features

# Prediction
y_pred = decoder_op     # After

# Targets(Labels) are the input data
y_true = X          # Before

# Define loss and optimizer, minimize the quared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initializing all the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)  ## max(x) =1 ,min(x) = 0  ---> so use sigmoid
    # Training cycle
    for epoch in range(train_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op(backprop) and cost op(to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            if epoch % display_step == 0:
                print("Epoch:", "%04d" % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")
        # Applyng encode and decode over test set
        encoder_decode = sess.run(
            y_pred, feed_dict={X: mnist.test.images[:example_to_show]})
        # Compare orginal images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(example_to_show):
            a[0][1].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encoder_decode[i], (28, 28)))
        plt.show()
