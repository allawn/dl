import tensorflow as tf
input = [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]]
out=tf.gather(input, [0, 1], axis=0)
sess = tf.InteractiveSession()
print(out.eval())