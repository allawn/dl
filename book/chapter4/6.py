import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
a = np.array([[1,2,3,4,5],[4,5,6,7,8],[9,10,11,12,13]])
print(tf.slice(a,[1,2],[-1,2]).eval())