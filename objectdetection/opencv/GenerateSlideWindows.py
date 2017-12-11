import cv2
import numpy as np
import numpy.random as npr
from PIL import Image, ImageDraw
from pylab import *

img = cv2.imread("0_Parade_marchingband_1_849.jpg")
height, width, channel = img.shape
size = npr.randint(12, min(width, height) / 2)
#top_left
nx = npr.randint(0, width - size)
ny = npr.randint(0, height - size)
#random crop
crop_box = np.array([nx, ny, nx + size, ny + size])

cropped_im = img[ny : ny + size, nx : nx + size, :]
resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("resize.jpeg", resized_im)

imshow(resized_im)
show()