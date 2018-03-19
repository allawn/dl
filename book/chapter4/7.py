import tensorflow as tf
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
t3=tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
t4=tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
sess = tf.InteractiveSession()
print(t3.eval())
print(t4.eval())