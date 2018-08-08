import tensorflow as tf
#�����������ݱ�ǩ�����������
label = tf.constant([0, 1, 2], dtype = tf.int32)
#�����������ݣ���Ϊ�����������ֵ
input = tf.constant([[0, 0], [1, 1], [2, 2]], dtype = tf.float32)
#����������ģ�ͣ�������ģ�����ֵΪ3������
logist = tf.layers.dense(input, 3)
#��������������ʧ����
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = logist)
#�����Ż��㷨
trainop = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
init = tf.global_variables_initializer()
#ִ��ģ��ѵ��
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(trainop)
        print(sess.run(logist))
