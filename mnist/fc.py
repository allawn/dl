import tensorflow as tf
#��������lableֵ
label = tf.constant([0, 0, 1, 1], dtype = tf.int32)
#��������ֵ��Ϊ�������ݣ�ÿ��������3��ֵ���ֱ��Ӧx_1, x_2, x_3
input = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype = tf.float32)
#��һ�����ز㣬���ĸ���Ԫ
hiddenlayer1 = tf.layers.dense(input, 4)
#�ڶ������ز㣬��������Ԫ
hiddenlayer2 = tf.layers.dense(hiddenlayer1, 3)
#����㣬�и�������Ԫ���ֱ��Ӧy_1, y_2
logist = tf.layers.dense(hiddenlayer2, 2)
#������ʧ���̣����ý�����
loss = tf.losses.sparse_softmax_cross_entropy(logits = logist, labels = label)
#����ѵ������
trainop = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()
#ִ��ģ��ѵ����������������һ��ļ���ֵ
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(trainop)
        print(sess.run(logist))
