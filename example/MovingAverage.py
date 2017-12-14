#shadow_variable = decay*shadow_variable+(1-decay)*variable
# 每次使用的衰减率 ＝ min{decay, (1+num_updates)/(10+unm_updates)}
#我们使用滑动平均模型可以使模型在测试数据上更准确。

import tensorflow as tf

#需要保存滑动平均值的变量
v1 = tf.Variable(0, dtype=tf.float32)
v2 = tf.Variable(0, dtype=tf.float32)

#步数
step = tf.Variable(0, trainable=False)
#滑动平均模型
ema = tf.train.ExponentialMovingAverage(0.99,step)
#向模型提供变量
averages_op = ema.apply([v1,v2])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print (sess.run([v1,ema.average(v1),v2,ema.average(v2)]))

    sess.run(tf.assign(v1,5))
    sess.run(tf.assign(v2,8))
#想获得影子变量，需要在run一下滑动平均节点
    sess.run(averages_op)
    print (sess.run([v1,ema.average(v1),v2,ema.average(v2)]))

