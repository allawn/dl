import tensorflow as tf
#样本数据lable值
label = tf.constant([0, 0, 1, 1], dtype = tf.int32)
#样本数据值，为输入数据，每个输入有3个值，分别对应x_1, x_2, x_3
input = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype = tf.float32)
#第一个隐藏层，有四个神经元
hiddenlayer1 = tf.layers.dense(input, 4)
#第二个隐藏层，有三个神经元
hiddenlayer2 = tf.layers.dense(hiddenlayer1, 3)
#输出层，有个两个神经元，分别对应y_1, y_2
logist = tf.layers.dense(hiddenlayer2, 2)
#定义损失方程，采用交叉熵
loss = tf.losses.sparse_softmax_cross_entropy(logits = logist, labels = label)
#定义训练操作
trainop = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()
#执行模型训练，输出神经网络最后一层的激活值
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(trainop)
        print(sess.run(logist))
