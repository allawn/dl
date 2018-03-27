import tensorflow.contrib.slim as slim
import tensorflow as tf

input = slim.model_variable('weights',
                              shape=[10, 28, 28 , 1],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
net = slim.repeat(input, 3, slim.conv2d, 256, [3, 3], scope='conv3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
net = slim.fully_connected(net, 32)
net=slim.flatten(net)
net=slim.stack(net, slim.fully_connected, [128, 32, 10], scope='fc')
print(net)
