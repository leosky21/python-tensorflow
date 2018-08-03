# coding: utf-8
import tensorflow as tf

'''
 generally , we use linear function instead of custom function
     define four parameters in function add_layer:
        -   inputs: the value to be input
        -   size: the size of the value
        -   out_size: the size of output value
        -   activation_function: the function of activation, we set it to use linear equations by default 
'''


def add_layer(inputs, in_size, out_size, activation_function=None):
    ## normal_distribution is better than zeros ##
    Layer_Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    ## donnot recommend zero ##
    biases = tf.Variable(tf.zeros([1., out_size]) + 0.1)

    ## the unactivatived value of the neural netework ##
    Wx_plus_b = tf.mamual(inputs, Layer_Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function
    return outputs
