import tensorflow as tf
import DistributedTensorFlowDemo.linearmodel as lr
import DistributedTensorFlowDemo.model_deploy as model_deploy
import time
import functools

FLAGS = tf.app.flags.FLAGS
max_steps=5000
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

tf.app.flags.DEFINE_integer('num_clones', 2, 'Number of clones to deploy per worker.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', True,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')

cluster = tf.train.ClusterSpec(
    {"worker": ["localhost:2222", "localhost:2223"], "ps": ["localhost:2224"]})



server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

def _create_losses(x,y):
    logist, _, _ = lr.inference(x)
    loss = lr.loss(logist, y)
    tf.losses.add_loss(loss)

if FLAGS.job_name == 'ps':
    server.join()

else:
    with tf.Graph().as_default():

        is_chief = (FLAGS.task_id == 0)
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task_id,
            num_replicas=2,
            num_ps_tasks=1,
            )

        with tf.device(deploy_config.variables_device()):
            global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer = tf.constant_initializer(0), trainable = False)

        with tf.device(deploy_config.inputs_device()):
            x = tf.placeholder(tf.float32)
            y = tf.placeholder(tf.float32)

        model_fn = functools.partial(_create_losses,x=x,y=y)
        clones = model_deploy.create_clones(deploy_config, model_fn)

        with tf.device(deploy_config.optimizer_device()):
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones, optimizer, regularization_losses=None)

            grad_updates = optimizer.apply_gradients(grads_and_vars,
                                                              global_step=global_step)

            update_ops=[]
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')


        x_train = [1, 2, 3, 4]
        y_train = [0, -1, -2, -3]
        stopHook=tf.train.StopAtStepHook( last_step=max_steps)


        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,hooks=[stopHook]) as sess:
            while not sess.should_stop():
                try:
                    cur_gl_step=sess.run(global_step)
                    print("curr_gl_step: ",cur_gl_step)
                    curr_loss=sess.run([train_tensor], feed_dict={x: x_train, y: y_train})
                    print("loss: %s" %(curr_loss))
                    if is_chief:
                        time.sleep(0.01)
                    else:
                        time.sleep(0.02)
                except RuntimeError:
                     print("deal Run called even after should_stop requested")
                     break





