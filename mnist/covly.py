import tensorflow as tf
import cv2
#通过opencv库，读取图片数据为数组，数组为维度信息为28 * 28
img = cv2.imread("1.png", 0)
#构建TensorFlow图的输入数据，features可以认为是图片的特征数据
features = tf.Variable(initial_value=img, dtype=tf.float32)

# 构建输入层，改变输入数据维度为 4-D tensor: [batch_size, width, height, channels]，图像数据为 28 * 28像素大小, 并且为单通道
input_layer = tf.reshape(features, [-1, 28, 28, 1])

# 构建卷积层1，卷积核大小为5 * 5，卷积核数量为32，激活函数使用relu；输入Tensor维度为[batch_size, 28, 28, 1]；输出Tensor维度为[batch_size, 28, 28, 32]
conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

# 构建池化层1，
# 采用2x2维度的最大化池化操作，步长为2
# 输入Tensor维度: [batch_size, 28, 28, 32]
# 输出Tenso维度: [batch_size, 14, 14, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 构建卷积层2，卷积核大小为5 * 5，卷积核数量为64，激活方法使用relu；输入Tensor维度为[batch_size, 14, 14, 32]；输出Tensor维度为[batch_size, 14, 14, 64]
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

# 构建池化层2，采用2 * 2维度的最大化池化操作，步长为2；输入Tensor维度为[batch_size, 14, 14, 64]；输出Tensor维度为[batch_size, 7, 7, 64]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 展开池化层2输出Tensor为一个向量；输入Tensor维度为[batch_size, 7, 7, 64]；输出Tensor维度为 [batch_size, 7 * 7 * 64]
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# 构建全链接层，该全链接层具有1024神经元，神经元使用relu激活函数；输入Tensor维度为[batch_size, 7 * 7 * 64]；输出Tensor维度为[batch_size, 1024]
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

# 对全链接层的数据加入dropout操作，防止过拟合；训练的时候，神经元会以40%的概率被dropout
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

# Logits层，对dropout层的输出Tensor，执行分类操作；输入Tensor维度维度为[batch_size, 1024]； 输出Tensor维度为[batch_size, 10]
logits = tf.layers.dense(inputs=dropout, units=10)
print(logits)
