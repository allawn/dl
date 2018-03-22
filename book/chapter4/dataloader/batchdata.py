import tensorflow as tf

tfrecord_filename = 'c:\\tmp\\book\\something.tfrecords'
reader = tf.TFRecordReader()
filenames = [tfrecord_filename]
filename_queue = tf.train.string_input_producer(filenames)
_, serialized_example = reader.read(filename_queue)
feature_set = { 'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64)
                }

features = tf.parse_single_example( serialized_example, features= feature_set )
image = features['image']
image=tf.decode_raw(image,tf.uint8)
image=tf.image.convert_image_dtype(image,tf.float32)
height = features['height']
height=tf.cast(height,tf.int32)
width = features['width']
width=tf.cast(width,tf.int32)
label = features['label']
label=tf.cast(label,tf.int32)
image=tf.reshape(image,shape=[height,width,3])
image=tf.image.resize_images(image,[300,300])
images,labels=tf.train.batch([image,label],batch_size=2)

with tf.Session() as sess:
    tf.train.start_queue_runners(sess)
    for i in range(10):
        imgsVal,lablesVal=sess.run([images,labels])
        print(imgsVal.shape)
        print(lablesVal)
