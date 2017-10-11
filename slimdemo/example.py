import tensorflow as tf
import tensorflow.contrib.slim as slim

weights = slim.variable('weights',
                        shape=[10, 10, 3 , 3],
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        regularizer=slim.l2_regularizer(0.05),
                        device='/CPU:0')

# Model Variables
weights = slim.model_variable('weights2',
                              shape=[10, 10, 3 , 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
model_variables = slim.get_model_variables()

# Regular variables
my_var = slim.variable('my_var',
                       shape=[20, 1],
                       initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()

var_aa=tf.get_variable("aaaVar",shape=[10,20])

my_model_variable = var_aa

# Letting TF-Slim know about the additional variable.
slim.add_model_variable(my_model_variable)

input=tf.get_variable("input_var",shape=[1,24,24,3],dtype=tf.float32)

net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')
print(net)

slim.repeat()