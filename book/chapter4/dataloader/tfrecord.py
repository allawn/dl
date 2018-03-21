import tensorflow as tf
import glob
from PIL import Image
import numpy as np

images = glob.glob('./*.jpg')
label=0

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = 'c:\\tmp\\something.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecord_filename)

for img in images:
    imgContent = Image.open(img)
    imgContent = np.array(imgContent)
    print(imgContent.shape,imgContent.dtype)
    #     imgContent = tf.read_file(img)
    feature = { 'label': _int64_feature(label),'image': _bytes_feature(imgContent.tostring())}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()

reader = tf.TFRecordReader()
filenames = glob.glob('*.tfrecords')
filename_queue = tf.train.string_input_producer(filenames)
_, serialized_example = reader.read(filename_queue)
feature_set = { 'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
                }

features = tf.parse_single_example( serialized_example, features= feature_set )
image = features['image']
label = features['label']

with tf.Session() as sess:
    tf.train.start_queue_runners(sess)
    imgstr,lable=sess.run([image,label])
    img=np.fromstring(imgstr,dtype=np.uint8)
    print(img)
    print(img.shape)
    print(lable)


