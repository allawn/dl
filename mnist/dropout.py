import tensorflow as tf
#定义数据占位符，用以接收dropout的概率
dropout = tf.placeholder(tf.float32)
#定义输入变量，用以模型神经元的输出，变量为5 *5 大小的矩阵，里面元素为1
x = tf.Variable(tf.ones([5, 5]))
#定义dropout操作
y = tf.nn.dropout(x, dropout)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#传入dropout概率0.4, 并运行dropout操作
print(sess.run(y, feed_dict={dropout: 0.4}))
