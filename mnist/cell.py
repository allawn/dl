import tensorflow as tf
#表示输入数据，数据维度为1 * 2大小的浮点数矩阵
x = tf.placeholder(tf.float32, [1, 2])

# 定义神经元中的权重系数和偏置，都为随机浮点数，其中权重系数矩阵维度为2 * 1，偏置维度为1
W = tf.Variable(tf.random_normal([2, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "biases")

# 定义神经元的激活输出，其中sigmoid为激活函数，tf.matmul(x, W) + b为输入数据的加权求和
y = tf.sigmoid(tf.matmul(x, W) + b)
#定义模型数据的初始化操作
init = tf.global_variables_initializer()
with tf.Session() as sess:
sess.run(init)
#输入神经元的数据为[[1, 1]]，神经元的输出为y
    print(sess.run(y, feed_dict = {x: [[1, 1]]}))
