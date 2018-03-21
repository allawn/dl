import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

image_raw = tf.gfile.FastGFile('167.png','rb').read()
image0 = tf.image.decode_image(image_raw)
print(image0)
with tf.Session() as sess:
    out=sess.run(image0)
    print(out.shape)
    print(out.dtype)
    img = Image.fromarray(out)
    # plt.imshow(out)
    # plt.show()