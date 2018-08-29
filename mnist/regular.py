import tensorflow as tf
from tensorflow.contrib import layers
#����L1���������ϵ��Ϊ0.1
regularizer1 = layers.l1_regularizer(0.1)
#����L2���������ϵ��Ϊ0.05
regularizer2 = layers.l2_regularizer(0.05)
#����ģ�͵�һ�������������ռ�Ϊvar/weight��shapeΪ[8]����ʼֵΪ1�����Ը�ģ�Ͳ�������L1������
with tf.variable_scope('var', initializer = tf.random_normal_initializer(), regularizer = regularizer1):
weight = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())
#����ģ�͵ڶ��������������ռ�Ϊvar2/weight��shapeΪ[8]����ʼֵΪ1�����Ը�ģ�Ͳ�������L2������
with tf.variable_scope('var2', initializer = tf.random_normal_initializer(), regularizer = regularizer2):
weight2 = tf.get_variable('weight', shape = [8], initializer = tf.ones_initializer())
#��ӡ���������а�����ģ�ͱ���
print(tf.get_collection(tf.GraphKeys.VARIABLES))
#���ģ�����򻯼����а�����������Ŀ
print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#������ʧ�����е�������ʧֵ��Ϊ���򻯼���������������Ŀ���ݵľۺ�ֵ
regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#��ӡģ���������Tensor����
print(regularization_loss)
