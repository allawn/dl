import tensorflow as tf
#定义样本数据标签，有三个类别
label = tf.constant([0, 1, 2], dtype = tf.int32)
#定义样本数据，作为神经网络的输入值
input = tf.constant([[0, 0], [1, 1], [2, 2]], dtype = tf.float32)
#定义神经网络模型，神经网络模型输出值为3个数据
logist = tf.layers.dense(input, 3)
#定义分类问题的损失方程
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = logist)
#定义优化算法
trainop = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
init = tf.global_variables_initializer()
#执行模型训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(trainop)
        print(sess.run(logist))
