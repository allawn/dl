import tensorflow as tf
#定义卷积操作的共享权重，卷积核的大小为5 * 5 * 1，卷积核个数为3；采用截断正态分布初始化
filter = tf.get_variable('weights', [5, 5, 1, 3], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
#第一输入数据，输入数据shape为[1, 20, 20, 1]，其中第一个1表示输入数据batchsize大小；20 * 20可以认为是输入数据的高度和宽度；第二个1可以认为是输入数据的通道个数；输入数据采用截断正态分布初始化
inputData = tf.get_variable(shape=[1, 20, 20, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer, name="inputData")
#输出输入数据信息
print(inputData)
#定义卷积操作，input表示数据数据，维度信息为[batch, in_height, in_width, in_channels]；filter表示卷积核，维度信息为[filter_height, filter_width, in_channels, out_channels]；strides为步长信息，[1, 2, 2, 1]表示在输入数据高度in_height维度的步长为2，宽度in_width维度的步长为2；padding表示数据的填存方式，SAME表示在需要的时候需要对数据进行填存，一般以0来填充。
cov1 = tf.nn.conv2d(input = inputData, filter = filter, strides = [1, 2, 2, 1], padding = "SAME")
print(cov1)
