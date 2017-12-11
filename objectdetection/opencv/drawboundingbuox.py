from PIL import Image, ImageDraw
from pylab import *

#annotation
#0--Parade/0_Parade_marchingband_1_849 448.51 329.63 570.09 478.23
annotation=[448.51,329.63,570.09,478.23]
im = Image.open("0_Parade_marchingband_1_849.jpg")
draw = ImageDraw.Draw(im)

line = 5
x, y = annotation[0], annotation[1]
width= annotation[2]-annotation[0]
height = annotation[3]-annotation[1]

for i in range(1, line + 1):
    draw.rectangle((x + (line - i), y + (line - i), x + width + i, y + height + i), outline='red')

# imshow(im)
# show()
im.save("out.jpeg")
