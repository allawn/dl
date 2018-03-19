#tensor变换

import tensorflow as tf

tensor=tf.ones(shape=[2,3,4],dtype=tf.float32,name="oneTensor")
reduce1=tf.reduce_sum(tensor,axis=0)
reduce2=tf.reduce_sum(tensor,axis=1)

with tf.Session() as sess:
    print(sess.run(reduce1))
    print(sess.run(reduce2))