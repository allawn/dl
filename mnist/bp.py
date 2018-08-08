import tensorflow as tf
# 表示输入数据，数据维度为1 * 2大小的浮点数矩阵
x = tf.placeholder(tf.float32, shape = [1, 2])
# 定义神经元中的权重系数和偏置，都为随机浮点数，其中权重系数矩阵维度为2 * 1，偏置维度为1
W = tf.Variable(tf.random_normal([2, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "biases")
# 定义神经元的激活输出，其中sigmoid为激活函数，tf.matmul(x, W) + b为输入数据的加权求和
y = tf.sigmoid(tf.matmul(x, W) + b)
#定义样本数据对应的目标值
target = 0
#定义优化方法，为了验证反向传播算法，把学习率设置为1
opt = tf.train.GradientDescentOptimizer(learning_rate = 1)
# 定义损失函数，根据公式的定义，损失函数对应网络输出值的导数为(y - target)
loss = 1 / 2 * (y - target) ** 2
#反向传播算法中，梯度自动计算
grads_and_vars = opt.compute_gradients(loss)
#使用梯度下降法，更新模型参数
applyGrads = opt.apply_gradients(grads_and_vars)
#收集模型中参数对应的梯度值，用于做比较验证反向传播算法
grads = []
for grad, var in grads_and_vars:
    grads.append(grad)
# 定义模型数据的初始化操作
init = tf.global_variables_initializer()
with tf.Session() as sess:
sess.run(init)
#计算模型loss值，神经网络输出值y，以及模型参数对应的梯度值，最后输出；x: [[1, 2]]为输入数据
    _, loss, y, grads = sess.run([applyGrads, loss, y, grads], feed_dict = {x: [[1, 2]]})
    print("loss:", loss)
    print("y:", y)
print("grads", grads)
