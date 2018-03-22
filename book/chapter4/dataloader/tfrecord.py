import tensorflow as tf
import glob
from PIL import Image
import numpy as np
import os

images = glob.glob('img/*/*.jpg')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = 'c:\\tmp\\book\\something.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecord_filename)

for img in images:
    imgPath=os.path.abspath(img)
    label=int(os.path.dirname(imgPath).split(os.path.sep)[-1])
    imgContent = Image.open(imgPath)
    imgContent = np.array(imgContent)
    height=imgContent.shape[0]
    width=imgContent.shape[1]
    feature = { 'label': _int64_feature(label), 'height':_int64_feature(height),'width':_int64_feature(width),'image': _bytes_feature(imgContent.tostring())}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()

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
height = features['height']
height=tf.cast(height,tf.int32)
width = features['width']
width=tf.cast(width,tf.int32)
label = features['label']
label=tf.cast(label,tf.int64)
image=tf.reshape(image,shape=[height,width,3])

with tf.Session() as sess:
    tf.train.start_queue_runners(sess)
    for i in range(100):
        imgVal,heightVal,widthVal,lableVal=sess.run([image,height,width,label])
        print(i)
        print(imgVal.shape)
        print(lableVal)
        print(heightVal)
        print(widthVal)


