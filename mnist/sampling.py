import tensorflow as tf
#定义输入数据
inputData = tf.get_variable(shape = [1, 20, 20, 1], dtype = tf.float32, name = "inputData")
print(inputData)
#定义卷积核
filter = tf.get_variable('weights', [5, 5, 1, 3], dtype = tf.float32)
#定义卷积操作
cov1 = tf.nn.conv2d(input = inputData, filter = filter, strides = [1, 2, 2, 1], padding = "SAME")
print(cov1)
#定义均值下采样操作，value表示输入数据，ksize为下采样操作核大小；strides为下采样操作步长
pool1 = tf.nn.avg_pool(value = cov1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
print(pool1)
