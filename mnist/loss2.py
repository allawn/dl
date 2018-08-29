import tensorflow as tf
#样本数据标签值
label = tf.constant([[0], [1], [2]], dtype = tf.float32)
#样本数据，神经网络的输入数据
input = tf.constant([[0, 0], [1, 1], [2, 2]], dtype = tf.float32)
#定义神经网络模型，模型输出值为1个
logist = tf.layers.dense(input, 1)
print(logist)
#定义损失方程，采用均方差损失函数
loss = tf.losses.mean_squared_error(labels = label, predictions = logist)
#定义模型优化算法
trainop = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
init = tf.global_variables_initializer()
#执行模型训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        sess.run(trainop)
        print(sess.run(logist))
