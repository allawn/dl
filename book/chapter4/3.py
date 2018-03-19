import tensorflow as tf

t1 = tf.ones(shape=[2,1,3],dtype=tf.float32)
t2=tf.squeeze(t1)

with tf.Session() as sess:
    print(sess.run(t1))
    print(sess.run(t2))