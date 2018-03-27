import tensorflow.contrib.slim as slim
import tensorflow as tf

weights = slim.model_variable('weights',
                              shape=[10, 10, 3 , 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')

my_var = slim.variable('my_var',
                       shape=[20, 1],
                       initializer=tf.zeros_initializer())

print("ModelVariable",slim.get_model_variables())
print("All Variable",slim.get_variables())
