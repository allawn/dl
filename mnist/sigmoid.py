import tensorflow as tf
#���峣������Ϊ�����������ֵ
x = tf.constant([1, 2, 3], dtype = tf.float32)
#����sigmoid�����
y = tf.sigmoid(x)
#�����������ݵļ���ֵ
with tf.Session() as sess:
print(sess.run(y))
