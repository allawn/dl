import tensorflow as tf
#�������ݱ�ǩֵ
label = tf.constant([[0], [1], [2]], dtype = tf.float32)
#�������ݣ����������������
input = tf.constant([[0, 0], [1, 1], [2, 2]], dtype = tf.float32)
#����������ģ�ͣ�ģ�����ֵΪ1��
logist = tf.layers.dense(input, 1)
print(logist)
#������ʧ���̣����þ�������ʧ����
loss = tf.losses.mean_squared_error(labels = label, predictions = logist)
#����ģ���Ż��㷨
trainop = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
init = tf.global_variables_initializer()
#ִ��ģ��ѵ��
with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        sess.run(trainop)
        print(sess.run(logist))
