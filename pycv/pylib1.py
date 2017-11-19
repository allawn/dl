from PIL import Image
from pylab import *

#open image
pil_im=Image.open('src.jpg')

#gray pic and save
#pil_im.convert('L').save('bbb.jpg')

#缩略图
#pil_im.thumbnail((128,128))
#pil_im.save("out.jpg")

#crop 和paste 图像
# box=(100,100,400,400)
# region=pil_im.crop(box)
# region=region.transpose(Image.ROTATE_180)
# pil_im.paste(region,box)
# pil_im.save("out.jpg")

#图像尺寸和旋转
# out=pil_im.resize((128,128))
# out=out.rotate(45)
# out.save("out.jpg")


#绘制图片
#imshow(pil_im)
#show()
