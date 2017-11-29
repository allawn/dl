import tensorflow as tf
import example.linearmodel as lr
import time

FLAGS = tf.app.flags.FLAGS
max_steps=5000
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

cluster = tf.train.ClusterSpec(
    {"worker": ["localhost:2222", "localhost:2223"], "ps": ["localhost:2224"]})

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

if FLAGS.job_name == 'ps':
    server.join()

else:
    is_chief = (FLAGS.task_id == 0)
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % FLAGS.task_id, cluster=cluster)):
        #global_step = tf.contrib.framework.get_or_create_global_step()
        global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer = tf.constant_initializer(0), trainable = False)
        with tf.device('/cpu:0'):
            x = tf.placeholder(tf.float32)
            y = tf.placeholder(tf.float32)
            logist,W,b=lr.inference(x)
            loss=lr.loss(logist,y)

            optimizer = tf.train.GradientDescentOptimizer(0.01)
            grads=optimizer.compute_gradients(loss)
            applyGradsOp=optimizer.apply_gradients(grads,global_step=global_step)

            x_train = [1, 2, 3, 4]
            y_train = [0, -1, -2, -3]

            with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,hooks=[tf.train.StopAtStepHook(last_step=max_steps)]) as sess:
                while not sess.should_stop():
                    curr_step,curr_W, curr_b, curr_loss=sess.run([global_step,W, b, loss], {x: x_train, y: y_train})
                    print("curr_step: %s, W: %s, b: %s, loss: %s" %(curr_step,curr_W, curr_b, curr_loss))
                    sess.run(applyGradsOp)
                    time.sleep(0.01)





