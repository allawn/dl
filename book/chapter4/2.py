import tensorflow as tf

t1 = tf.ones(shape=[2,3],dtype=tf.float32)
t2 = tf.ones(shape=[2,3],dtype=tf.float32)

# 将t1, t2进行concat，axis为0，等价于将shape=[2, 2, 3]的Tensor concat成
#shape=[4, 3]的tensor。在新生成的Tensor中tensor[:2,:]代表之前的t1
#tensor[2:,:]是之前的t2
t3=tf.concat([t1, t2], 0)

# 将t1, t2进行concat，axis为1，等价于将shape=[2, 2, 3]的Tensor concat成
#shape=[2, 6]的tensor。在新生成的Tensor中tensor[:,:3]代表之前的t1
#tensor[:,3:]是之前的t2
t4=tf.concat([t1, t2], 1)

with tf.Session() as sess:
    print(sess.run(t3))
    print(sess.run(t4))