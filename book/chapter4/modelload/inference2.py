import tensorflow as tf
import os
import cv2

checkpointpath='C:\\tmp\\mnistckp'
def run(imgName):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(checkpointpath,'model.ckp.meta'))
        new_saver.restore(sess, checkpointpath)
        imgData=cv2.imread(imgName,0)
        input=tf.convert_to_tensor(imgData,dtype=tf.float32)

if __name__ == '__main__':
    imgName="1.png"
    run(imgName)