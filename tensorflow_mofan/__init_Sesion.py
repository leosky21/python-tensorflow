import tensorflow as tf
# session has two ways to be opened

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2]
                       , [2]])
product = tf.matmul(matrix1, matrix2) ## matrix multiply np.dot(m1,m2)

## method fir

with tf.Session() as s:   ## we open this sesson , and please python help me to close it
    result = s.run(product)
    print (result)

## method sec
sess = tf.Session()
result2 = sess.run(product)
print (result2)
sess.close()