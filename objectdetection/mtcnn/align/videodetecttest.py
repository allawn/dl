import numpy as np
import cv2
import  tensorflow as tf
import detect_face
import time

cap = cv2.VideoCapture(0)
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '.')
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            bounding_boxes, points = detect_face.detect_face(frame, 24, pnet, rnet, onet, threshold, factor)
            # print(points)
            img_color = frame
            for face_position in bounding_boxes:
                face_position = face_position.astype(int)
                print(face_position[0:4])
                cv2.rectangle(img_color, (face_position[0], face_position[1]), (face_position[2], face_position[3]),
                              (0, 255, 0), 2)
                # cv2.circle(img_color, (points[0], points[1]), 5, (0, 255, 0),-1)
                # cv2.circle(img_color, (points[2], points[3]), 5, (0, 255, 0), -1)
                # cv2.circle(img_color, (points[4], points[5]), 5, (0, 255, 0), -1)
                # cv2.circle(img_color, (points[6], points[7]), 5, (0, 255, 0), -1)
                # cv2.circle(img_color, (points[8], points[9]), 5, (0, 255, 0), -1)



            # Our operations on the frame come here
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            # Display the resulting frame
            cv2.imshow('frame',img_color)
            time.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()