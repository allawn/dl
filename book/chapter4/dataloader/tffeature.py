import tensorflow as tf

tfrecord_filename = 'out.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecord_filename)
feature={
    'age':tf.train.Feature(int64_list=tf.train.Int64List(value=[30])),
    'name' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes('wuwei',encoding='utf-8')]))
         }
example=tf.train.Example(features=tf.train.Features(feature=feature))
writer.write(example.SerializeToString())
writer.close()

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer([tfrecord_filename])
_, serialized_example = reader.read(filename_queue)
feature_set={
    'name': tf.FixedLenFeature([], tf.string),
    'age': tf.FixedLenFeature([], tf.int64)
    }  
features = tf.parse_single_example( serialized_example, features= feature_set )
name = features['name']
age = features['age']
with tf.Session() as sess:
  tf.train.start_queue_runners(sess)
  nameVal,ageVal=sess.run([name,age])
  print(nameVal,ageVal)
