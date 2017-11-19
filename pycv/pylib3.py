from PIL import Image
from pylab import *

im=array(Image.open("src.jpg").convert("L"))
#新建图
figure()

#轮廓
gray()
contour(im, origin='image')
axis('equal')
axis('off')

#新建另外一个
figure()
#直方图
hist(im.flatten(),128)

show()