import tensorflow as tf
from tensorflow.contrib import layers
#定义L1正则化项，正则化系数为0.1
regularizer1 = layers.l1_regularizer(0.1)
#定义L2正则化项，正则化系数为0.05
regularizer2 = layers.l2_regularizer(0.05)
#定义模型第一个参数，命名空间为var/weight，shape为[8]，初始值为1，并对改模型参数加入L1正则化项
with tf.variable_scope('var', initializer = tf.random_normal_initializer(), regularizer = regularizer1):
weight = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())
#定义模型第二个参数，命名空间为var2/weight，shape为[8]，初始值为1，并对改模型参数加入L2正则化项
with tf.variable_scope('var2', initializer = tf.random_normal_initializer(), regularizer = regularizer2):
weight2 = tf.get_variable('weight', shape = [8], initializer = tf.ones_initializer())
#打印变量集合中包含的模型变量
print(tf.get_collection(tf.GraphKeys.VARIABLES))
#输出模型正则化集合中包含的正则化项目
print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#定义损失方程中的正则化损失值，为正则化集合中所有正则化项目数据的聚合值
regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#打印模型正则化项的Tensor定义
print(regularization_loss)
