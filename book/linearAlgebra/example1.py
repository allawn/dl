import tensorflow as tf

scalarVar = tf.Variable(initial_value=0.5, dtype=tf.float32)
vectorVar = tf.Variable(initial_value=tf.zeros(shape=[10]),dtype=tf.float32)
matrixVar = tf.Variable(initial_value=tf.ones(shape=[10,10]), dtype=tf.float32)
tensorVar = tf.Variable(initial_value=tf.ones(shape=[10,10,10]), dtype=tf.float32)

matrix_A = tf.Variable(initial_value=[[2,2],[3,3]], dtype=tf.float32)
vector_b = tf.Variable(initial_value=[[4],[4]],dtype=tf.float32)

# vector_c = tf.reduce_sum(tf.matmul(matrix_A, vector_b),axis=1)
vector_c=tf.matmul(matrix_A,vector_b)


matrix_A = tf.Variable(initial_value=[[1,1],[2,2]], dtype=tf.float32)
matrix_B = tf.Variable(initial_value=[[3,3],[4,4]], dtype=tf.float32)
matrix_C = tf.matmul(matrix_A,matrix_B)

session=tf.Session()
initOP=tf.global_variables_initializer()
session.run(initOP)

# print(scalarVar.eval(session))
# print(vectorVar.eval(session))
# print(matrixVar.eval(session))
# print(tensorVar.eval(session))
print(session.run(vector_c))

