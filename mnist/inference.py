import tensorflow as tf
import train
import cv2
checkPointPath="C:\\tmp\\mnistckp\\model.ckp"

def run(imgName):

    graph = tf.Graph()
    with graph.as_default():
        imgData=cv2.imread(imgName,0)
        input=tf.convert_to_tensor(imgData,dtype=tf.float32)
        logist=train.inference(input)
        prediction = tf.nn.softmax(logist)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess,checkPointPath)
        predictionVal=sess.run(prediction)
        top_1 = predictionVal.argsort()[0][::-1][0]

        print("The prediction val is: ",predictionVal)
        print("The predict class is: {}".format(top_1))



if __name__ == '__main__':
    imgName="1.png"
    run(imgName)