import tensorflow as tf
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.nets as nets

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

#tf.device()
#tf.placeholder()


slim.repeat()

def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net


vgg=nets.vgg
images=tf.placeholder(dtype=tf.float32,shape=[1,32,32,3])
labels=tf.placeholder(dtype=tf.int32,shape=[1])
predictions, _=vgg.vgg_16(images)
loss = slim.losses.softmax_cross_entropy(predictions, labels)
sum_of_squares_loss = slim.losses.sum_of_squares(predictions, labels)
slim.losses.add_loss(sum_of_squares_loss)
total_loss2 = slim.losses.get_total_loss()

total_loss = slim.losses.get_total_loss()
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = slim.learning.create_train_op(total_loss, optimizer)
logdir =""

slim.learning.train(
    train_op,
    logdir,
    number_of_steps=1000,
    save_summaries_secs=300,
    save_interval_secs=600)

variables_to_restore = slim.get_variables_by_name("v2")
# or
variables_to_restore = slim.get_variables_by_suffix("2")
# or
variables_to_restore = slim.get_variables(scope="nested")
# or
variables_to_restore = slim.get_variables_to_restore(include=["nested"])
# or
variables_to_restore = slim.get_variables_to_restore(exclude=["v1"])

restorer = tf.train.Saver(variables_to_restore)
with tf.Session as sess:
  restorer.restore(sess, "/tmp/model.ckpt")


# Load the data
images, labels = None,None

# Define the network
predictions = None

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    'accuracy': slim.metrics.accuracy(predictions, labels),
    'precision': slim.metrics.precision(predictions, labels),
    'recall': slim.metrics.recall(mean_relative_errors, 0.3),
})

# Create the summary ops such that they also print out to std output:
summary_ops = []
for metric_name, metric_value in names_to_values.iteritems():
    op = tf.summary.scalar(metric_name, metric_value)
    op = tf.Print(op, [metric_value], metric_name)
    summary_ops.append(op)

num_examples = 10000
batch_size = 32
num_batches = tf.math.ceil(num_examples / float(batch_size))

# Setup the global step.
slim.get_or_create_global_step()

slim.evaluation.evaluation_loop(
    'local',
    "",
    "",
    num_evals=num_batches,
    eval_op=names_to_updates.values(),
    summary_op=tf.summary.merge(summary_ops),
    eval_interval_secs=100)

images, labels = None,None

# Define the network
predictions = vgg.vgg_16(images)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
    "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
})

# Evaluate the model using 1000 batches of data:
num_batches = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for batch_id in range(num_batches):
        sess.run(names_to_updates.values())

    metric_values = sess.run(names_to_values.values())
    for metric, value in zip(names_to_values.keys(), metric_values):
        print('Metric %s has value: %f' % (metric, value))


#read data, using slim to read data
# _, string_tensor = slim.data.parallel_reader.parallel_read(
#     config.input_path,
#     reader_class=tf.TFRecordReader,
#     num_epochs=(input_reader_config.num_epochs
#                 if input_reader_config.num_epochs else None),
#     num_readers=input_reader_config.num_readers,
#     shuffle=input_reader_config.shuffle,
#     dtypes=[tf.string, tf.string],
#     capacity=input_reader_config.queue_capacity,
#     min_after_dequeue=input_reader_config.min_after_dequeue)

tf.image.decode_image() #aaa
tf.placeholder(tf.bool) #aaa
tf.constant(dtype=tf.bool,value=True)
tf.logging.set_verbosity(tf.logging.)
tf.Variable().op.name
tf.train.start_queue_runners(sess=)
tf.get_collection("aa")
xx=tf.train.Saver()
xx.save()
tf.Session()
# with tf.name_scope("train"):
#     bn = tf.layers.batch_normalization(
#         input_layer, fused=True, data_format='NCHW')
#
#     tf.contrib.layers.batch_norm(input_layer, fused=True, data_format='NCHW')

aa=tf.train.Server().target
tf.constant("0.5",tf.float32)
tf.Session(target=)
tf.get_collection("aa")
tf.add()
tf.train.Saver().save()

numWorkers=2
with tf.device("/job:ps/task:0/cpu:0"):
    w=tf.Variable(...)
    b=tf.Variable(...)

inputs=tf.split(0,numWorkers,input)
outputs=[]
for i in range(numWorkers):
    with tf.device("/job:worker/task:%d/gpu:0" % i):
        outputs.append(tf.matmul(input,w)+b)
loss=f(outputs)

with tf.device("/job:worker/task:0/gpu:0"):
    output=tf.matmul(input,w)+b
    loss=f(output)

with tf.device("/job:ps/task:0/cpu:0"):
    w=tf.Variable(...)
    b=tf.Variable(...)
with tf.device("/job:worker/task:0/gpu:0"):
    output=tf.matmul(input,w)+b
    loss=f(output)

with tf.device("/job:ps/task:0/cpu:0"):
    w=tf.Variable(...)
    b=tf.Variable(...)
with tf.device("/job:worker/task:1/gpu:0"):
    output=tf.matmul(input,w)+b
    loss=f(output)

with tf.device(tf.train.replica_device_setter(ps_tasks=2)):
    w1=tf.Variable(...)
    b1=tf.Variable(...)
    w2=tf.Variable(...)
    b2=tf.Variable(...)
    ...
    ...