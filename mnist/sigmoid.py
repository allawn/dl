import tensorflow as tf
#定义常量，作为激活函数的输入值
x = tf.constant([1, 2, 3], dtype = tf.float32)
#定义sigmoid激活函数
y = tf.sigmoid(x)
#计算输入数据的激活值
with tf.Session() as sess:
print(sess.run(y))
