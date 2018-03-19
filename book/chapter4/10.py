import tensorflow as tf
t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1], [2, 2]])
out1=tf.pad(t, paddings, "CONSTANT")    # [[0, 0, 0, 0, 0, 0, 0],
                                        #  [0, 0, 1, 2, 3, 0, 0],
                                        #  [0, 0, 4, 5, 6, 0, 0],
                                        #  [0, 0, 0, 0, 0, 0, 0]]

out2=tf.pad(t, paddings, "REFLECT")     # [[6, 5, 4, 5, 6, 5, 4],
                                        #  [3, 2, 1, 2, 3, 2, 1],
                                        #  [6, 5, 4, 5, 6, 5, 4],
                                        #  [3, 2, 1, 2, 3, 2, 1]]

out3=tf.pad(t, paddings, "SYMMETRIC")   # [[2, 1, 1, 2, 3, 3, 2],
                                        #  [2, 1, 1, 2, 3, 3, 2],
                                        #  [5, 4, 4, 5, 6, 6, 5],
                                        #  [5, 4, 4, 5, 6, 6, 5]]
sess = tf.InteractiveSession()
print(out1.eval())
print(out2.eval())
print(out3.eval())