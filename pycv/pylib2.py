from PIL import Image
from pylab import *

im=array(Image.open("src.jpg"))
#绘制图片
imshow(im)

x=[100,100,400,400]
y=[200,500,200,500]
#绘制点
plot(x,y,"r*")
print(x[:2])
#绘制线
plot(x[:2],y[:2])
title("aaaaaaaaaa")
axis('off')
show()