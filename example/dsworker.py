import sys
task_number = int(sys.argv[1])

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=task_number)

print("Starting server #{}".format(task_number))

server.start()
server.join()