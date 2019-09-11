import IPython.display
import PIL.Image
import numpy as np

# A simple function to display an image in an ipython notebook
def nbimage(data):
    IPython.display.display(PIL.Image.fromarray(data))

# create an image consisting of random colors

IMG_WIDTH = 512
IMG_HEIGHT = 288

data = np.zeros((IMG_HEIGHT,IMG_WIDTH,3), np.uint8) # a 512x512 image, with 3 color channels (R,G,B)

num_squares = np.random.randint(1, 100)

for i in range(num_squares):
    w = np.random.randint(2, IMG_WIDTH)
    h = np.random.randint(2, IMG_HEIGHT)

    x1 = np.random.randint(IMG_WIDTH-w)
    y1 = np.random.randint(IMG_HEIGHT-h)
    x2 = np.random.randint(x1+1, IMG_WIDTH)
    y2 = np.random.randint(y1+1, IMG_HEIGHT)

    rgb = np.random.randint(255,size=3)

    for i in range (3):
        layer = data[...,i]
        layer[...,y1:y2, x1:x2] = rgb[i]
        pixels = layer[...,y1:y2, x1:x2]
        pixels.fill(rgb[i])

# display it!
nbimage(data)