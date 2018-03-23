import tensorflow as tf
import  cv2
import numpy as np

def run(imgName):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        freeGraphPB = 'C:\\tmp\\mnistckp\\freezemodel.pb'
        graph_def = tf.GraphDef()
        f = tf.gfile.FastGFile(freeGraphPB, 'rb')
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def=graph_def)
        f.close()
        opname=[op.name for op in tf.get_default_graph().get_operations() ]
        print(opname)
        imgData = cv2.imread(imgName, 0)
        imgData = np.reshape(imgData, [-1, 28, 28, 1])
        imgData = imgData.astype(np.float32)
        inputdataname = graph.get_operation_by_name("import/inputdataName").outputs[0]
        inputLayer = graph.get_operation_by_name("import/inputLayer").outputs[0]
        predictionName = graph.get_operation_by_name("import/predictionName").outputs[0]
        predictionVal = sess.run(predictionName, feed_dict={inputLayer: imgData})
        top_1 = predictionVal.argsort()[0][::-1][0]

        print("The prediction val is: ",predictionVal)
        print("The predict class is: {}".format(top_1))


if __name__ == '__main__':
    imgName="117.png"
    run(imgName)