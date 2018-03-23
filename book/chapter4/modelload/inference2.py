import tensorflow as tf
import os
import cv2
import numpy as np

checkpointpath='C:\\tmp\\mnistckp'
def run(imgName):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(checkpointpath,'model.ckp.meta'))
        new_saver.restore(sess, checkpointpath)
        imgData=cv2.imread(imgName,0)
        imgData=np.reshape(imgData, [-1,28,28,1])
        imgData=imgData.astype(np.float32)
        graph = tf.get_default_graph()
        inputdataname=graph.get_operation_by_name("inputdataName").outputs[0]
        inputLayer=graph.get_operation_by_name("inputLayer").outputs[0]
        predictionName=graph.get_operation_by_name("predictionName").outputs[0]
        predictionVal=sess.run(predictionName,feed_dict={inputLayer:imgData})
        top_1 = predictionVal.argsort()[0][::-1][0]

        print("The prediction val is: ",predictionVal)
        print("The predict class is: {}".format(top_1))


if __name__ == '__main__':
    imgName="117.png"
    run(imgName)