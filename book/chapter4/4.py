import tensorflow as tf

t1 = tf.ones(shape=[2,4],dtype=tf.float32)
t2=tf.split(value=t1,num_or_size_splits=2,axis=1)

with tf.Session() as sess:
    print(sess.run(t1))
    print(sess.run(t2))