# coding: utf-8
"""Aim :  test  principle of data_reading of TensorFlow

   - Run : open Terminal and 'cd' to current dir and  'python3  __test_Read'
   - version:
        - tensorflow 1.8.0
        - python 3.6.5

    string_input_producer() --> wholeReader().read(queue_picList) --> local_variable_initializer --> start_queu_runner --> f.write(session.run())

"""

import tensorflow as tf

with tf.Session() as sess:
    # files to be read
    fileName = ['A.jpg', 'B.jpg', 'C.jpg']
    ''' to create a queue for the list of pics to be read
        ===============================================================
        list: list of files
        shuffle: 
            False means that the order of file_queue won't be disrupted
            True means that the order of file_queue should be disrupted
        num_epochs: 
           the number of list of pics to be put into file_queue in a sequence 
           if there are three pictures , it will be 15 pics
    '''
    fileName_queue = tf.train.string_input_producer(fileName, shuffle=False, num_epochs=5)
    ''' read data from queue of fileName. Corresponding method is reader.read()
        ======================================================================
        queue: the queue which is be translated by string_input_producer with list of filename
        name : default None 
    '''
    reader = tf.WholeFileReader()
    key, value = reader.read(fileName_queue)

    """from variables.py
    ======================================================================
    Returns an Op that initializes all local variables.

      This is just a shortcut for `variables_initializer(local_variables())`
    
      Returns:
        An Op that initializes all local variables in the graph.
    ======================================================================
    if initialize by mistake , python will throws this error 
    Exception in QueueRunner: 
        Attempting to use uninitialized value input_producer/limit_epochs/epochs
    """
    tf.initializers.local_variables().run()
    # tf.local_variables_initializer().run()
    '''
        string_input_producer just prepare for running, but not do it ;
        start_queue_runners start to do it in sess
        =================================================================
        sess=s : default None , if None ,run it in local session ,else in s
    '''
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    '''
    when run after five times ,  string_input_producer will tell python to throw a exception :'outOfRangeError'
    it doesn't matter , because string_input_producer set it at the end of the queue of the pic
    '''
    while True:
        i += 1
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
