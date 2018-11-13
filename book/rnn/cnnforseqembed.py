import tensorflow as tf
embedding_size=50
batchsize=32
maxlength=10

embedding=tf.ones(shape=[batchsize, maxlength, embedding_size])
embedding = tf.expand_dims(embedding, -1)  #for rank=4
seqWindows = 3
kernel_size = [seqWindows, embedding_size]
strides = [1, 1]
kernelNum1 = 256
net = tf.layers.conv2d(inputs=embedding, name="conv1", filters=kernelNum1,
                       filter=kernel_size, strides=strides,
                       padding="valid", activation=tf.nn.relu)

pool_size = [maxlength - seqWindows + 1, 1]
strides = [1, 1]
net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size, strides=strides, padding="valid",
                              name= "pool1")
net = tf.reshape(net, [-1, kernelNum1])