import tensorflow as tf
#������������
inputData = tf.get_variable(shape = [1, 20, 20, 1], dtype = tf.float32, name = "inputData")
print(inputData)
#��������
filter = tf.get_variable('weights', [5, 5, 1, 3], dtype = tf.float32)
#����������
cov1 = tf.nn.conv2d(input = inputData, filter = filter, strides = [1, 2, 2, 1], padding = "SAME")
print(cov1)
#�����ֵ�²���������value��ʾ�������ݣ�ksizeΪ�²��������˴�С��stridesΪ�²�����������
pool1 = tf.nn.avg_pool(value = cov1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
print(pool1)
