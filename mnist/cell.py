import tensorflow as tf
#��ʾ�������ݣ�����ά��Ϊ1 * 2��С�ĸ���������
x = tf.placeholder(tf.float32, [1, 2])

# ������Ԫ�е�Ȩ��ϵ����ƫ�ã���Ϊ���������������Ȩ��ϵ������ά��Ϊ2 * 1��ƫ��ά��Ϊ1
W = tf.Variable(tf.random_normal([2, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "biases")

# ������Ԫ�ļ������������sigmoidΪ�������tf.matmul(x, W) + bΪ�������ݵļ�Ȩ���
y = tf.sigmoid(tf.matmul(x, W) + b)
#����ģ�����ݵĳ�ʼ������
init = tf.global_variables_initializer()
with tf.Session() as sess:
sess.run(init)
#������Ԫ������Ϊ[[1, 1]]����Ԫ�����Ϊy
    print(sess.run(y, feed_dict = {x: [[1, 1]]}))
