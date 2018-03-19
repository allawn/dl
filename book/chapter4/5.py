import tensorflow as tf

t1 = tf.ones(shape=[2,4],dtype=tf.float32)
t2=tf.expand_dims(t1,axis=0)
shape=tf.shape(t2)
with tf.Session() as sess:
    print(sess.run(t1))
    print(sess.run(shape))
    print(sess.run(t2))