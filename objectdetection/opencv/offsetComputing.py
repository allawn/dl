import cv2
import numpy as np
import numpy.random as npr
from PIL import Image, ImageDraw
from pylab import *

img = cv2.imread("0_Parade_marchingband_1_849.jpg")
height, width, channel = img.shape
annotation=[448.51,329.63,570.09,478.23]
x1, y1, x2, y2 = annotation
w = x2 - x1 + 1
h = y2 - y1 + 1

size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

delta_x = npr.randint(-w * 0.2, w * 0.2)
delta_y = npr.randint(-h * 0.2, h * 0.2)

nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
nx2 = nx1 + size
ny2 = ny1 + size

crop_box = np.array([nx1, ny1, nx2, ny2])
offset_x1 = (x1 - nx1) / float(size)
offset_y1 = (y1 - ny1) / float(size)
offset_x2 = (x2 - nx2) / float(size)
offset_y2 = (y2 - ny2) / float(size)

cropped_im = img[ny1 : ny2, nx1 : nx2, :]
resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

imshow(resized_im)
show()