import IPython.display
import PIL.Image
import numpy as np
import random
import cv2

# A simple function to display an image in an ipython notebook
def nbimage(data):
    IPython.display.display(PIL.Image.fromarray(data))

# create an image consisting of random colors

IMG_WIDTH = 288
IMG_HEIGHT = 512

data = np.zeros((IMG_WIDTH,IMG_HEIGHT,3), np.uint8) # a 512x512 image, with 3 color channels (R,G,B)

# num_squares = np.random.randint(1, 3)
num_squares = 1

for i in range(num_squares):
    x1 = np.random.randint(IMG_WIDTH-1)
    y1 = np.random.randint(IMG_HEIGHT-1)
    x2 = np.random.randint(x1, IMG_WIDTH)
    y2 = np.random.randint(y1, IMG_HEIGHT)

    rgb = np.random.randint(255,size=3)
    print(rgb)

    data[x1:x2,y1][:,y1] = rgb
#     data[:,y1,x1:x2][:,y1] = rgb
#     data[:,y2,x1:x2][:,y2] = rgb 
print(data)
# by default, rand creates floating point numbers between [0,1].  We need to convert that to 8-bit bytes between [0,255]
# data = (255*data).astype(np.uint8)


# display it!
nbimage(data)