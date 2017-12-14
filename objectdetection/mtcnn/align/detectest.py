import  tensorflow as tf
import detect_face
import cv2
import matplotlib.pyplot as plt

print('Creating networks and loading parameters')
gpu_memory_fraction = 1.0
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '.')

img=cv2.imread('test2.jpg')
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

bounding_boxes, points = detect_face.detect_face(img, 24, pnet, rnet, onet, threshold, factor)
nrof_faces = bounding_boxes.shape[0]  # 人脸数目
print('找到人脸数目为：{}'.format(nrof_faces))
print(points)

img_color=img
crop_faces = []
for face_position in bounding_boxes:
    face_position = face_position.astype(int)
    print(face_position[0:4])
    cv2.rectangle(img_color, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
    crop = img_color[face_position[1]:face_position[3],
           face_position[0]:face_position[2], ]

    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)

    print(crop.shape)
    crop_faces.append(crop)
    plt.imshow(crop)
    plt.show()

plt.imshow(img_color)
plt.show()