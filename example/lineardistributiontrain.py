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

def create_done_queue(i):
    """Queue used to signal death for i'th ps shard. Intended to have
    all workers enqueue an item onto it to signal doneness."""

    with tf.device("/job:ps/task:%d" % (i)):
        return tf.FIFOQueue(2, tf.int32, shared_name="done_queue"+
                                                                 str(i))

def create_done_queues():
    return [create_done_queue(i) for i in range(1)]


if FLAGS.job_name == 'ps':
    # server.join()

    #auto stop server
    sess = tf.Session(server.target)
    queue = create_done_queue(0)

    # wait until all workers are done
    for i in range(2):
        sess.run(queue.dequeue())
        print("ps %d received worker %d done " % (0, i))

    print("ps %d: quitting"%(0))

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
        stopHook=tf.train.StopAtStepHook( last_step=max_steps)

        enq_ops = []
        for q in create_done_queues():
            qop = q.enqueue(1)
            enq_ops.append(qop)

        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,hooks=[stopHook]) as sess:
            curr_step=0
            while not sess.should_stop() and (curr_step+1)<max_steps:
                curr_step,curr_W, curr_b, curr_loss=sess.run([global_step,W, b, loss], feed_dict={x: x_train, y: y_train})
                print("curr_gl_step: %s, W: %s, b: %s, loss: %s" %(curr_step,curr_W, curr_b, curr_loss))
                sess.run(applyGradsOp,feed_dict={x: x_train, y: y_train})
                if is_chief:
                    time.sleep(0.01)
                else:
                    time.sleep(0.05)

            for op in enq_ops:
                sess.run(op)





