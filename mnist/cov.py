import tensorflow as tf
#�����������Ĺ���Ȩ�أ�����˵Ĵ�СΪ5 * 5 * 1������˸���Ϊ3�����ýض���̬�ֲ���ʼ��
filter = tf.get_variable('weights', [5, 5, 1, 3], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
#��һ�������ݣ���������shapeΪ[1, 20, 20, 1]�����е�һ��1��ʾ��������batchsize��С��20 * 20������Ϊ���������ݵĸ߶ȺͿ�ȣ��ڶ���1������Ϊ���������ݵ�ͨ���������������ݲ��ýض���̬�ֲ���ʼ��
inputData = tf.get_variable(shape=[1, 20, 20, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer, name="inputData")
#�������������Ϣ
print(inputData)
#������������input��ʾ�������ݣ�ά����ϢΪ[batch, in_height, in_width, in_channels]��filter��ʾ����ˣ�ά����ϢΪ[filter_height, filter_width, in_channels, out_channels]��stridesΪ������Ϣ��[1, 2, 2, 1]��ʾ���������ݸ߶�in_heightά�ȵĲ���Ϊ2�����in_widthά�ȵĲ���Ϊ2��padding��ʾ���ݵ���淽ʽ��SAME��ʾ����Ҫ��ʱ����Ҫ�����ݽ�����棬һ����0����䡣
cov1 = tf.nn.conv2d(input = inputData, filter = filter, strides = [1, 2, 2, 1], padding = "SAME")
print(cov1)
