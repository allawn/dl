import tensorflow as tf

sess=tf.Session()
# tf.train.start_queue_runners(sess=sess)

aa=tf.constant(0.5,tf.float32)
print(aa)

# for native train process.

#
# import threading
# from tensorflow.python.framework import ops
#
# gpu_options = tf.GPUOptions(allow_growth=True)
# config=tf.ConfigProto(gpu_options=gpu_options,device_count = {'GPU': 0})
# sess=tf.Session(config=config)
# init=tf.initialize_variables(tf.all_variables(), name="init//all_vars")
# sess.run(init)
#
# def _run(sess, enqueue_op):
#     while True:
#         sess.run(enqueue_op)
#
# collection=ops.GraphKeys.QUEUE_RUNNERS
# for qr in ops.get_collection(collection):
#     ret_threads = [threading.Thread(target=_run, args=(sess, op))
#                    for op in qr._enqueue_ops]
#     for t in ret_threads:
#         t.daemon = True
#         t.start()
#
# curIter=0
# # saver = tf.train.Saver()
# # saver.restore(sess,restore_path)
# for i in range(1000):
#     lossval=sess.run(tr_loss)
#     acc=sess.run(tr_accuracy)
#     elossval=sess.run(ts_loss)
#     eacc=sess.run(ts_accuracy)
#     gradients_and_vars=sess.run(grads)
#     sess.run(apply_gradient_op)
#     curIter = curIter+1
#     print("curIter:"+str(curIter)+", lossVal:"+str(lossval)+", accuracy:"+str(acc)+", eloss:"+str(elossval)+", eaccu:"+str(eacc))
