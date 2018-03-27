import tensorflow.contrib.slim as slim
import tensorflow as tf


inputs = slim.model_variable('weights',
                              shape=[10, 28, 28 , 1],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
  with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
    net = slim.conv2d(inputs, 64, [11, 11], stride=4, padding='VALID', scope='conv1')
    print(net)
    net = slim.conv2d(net, 256, [5, 5],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                      scope='conv2')
    print(net)
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')
    
print(net)    