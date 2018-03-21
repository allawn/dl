import tensorflow as tf

data=["aa","bb","cc"]
label=[1,2,3]

dataset=tf.data.Dataset.from_tensor_slices((data,label))
dataset=dataset.map(lambda x,y: (y,y))
dataset=dataset.repeat()
dataset=dataset.shuffle(10)
dataset=dataset.batch(2)
# iterator=dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()
tdata,tlabelnext_element = iterator.get_next()
print(tdata)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(10):
        print(sess.run(tdata))