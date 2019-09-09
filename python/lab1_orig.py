import IPython.display
import PIL.Image
import numpy as np
import random
import cv2

# A simple function to display an image in an ipython notebook
def nbimage(data):
    IPython.display.display(PIL.Image.fromarray(data))

def draw_square(data):
    

# create an image consisting of random colors

IMG_HEIGHT = 288
IMG_WIDTH = 512

data_zeros = np.zeros((IMG_HEIGHT,IMG_WIDTH,3), np.uint8) # a 512x288 image, with 3 color channels (R,G,B)
data = np.random.rand(IMG_HEIGHT,IMG_WIDTH,3)
print(data.shape)

# by default, rand creates floating point numbers between [0,1].  We need to convert that to 8-bit bytes between [0,255]
data = (255*data).astype(np.uint8)
print(data.shape)

# display it!
nbimage(data)