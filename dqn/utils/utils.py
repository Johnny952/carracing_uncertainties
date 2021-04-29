import numpy as np

def imgstackRGB2graystack(imgstack, R=0.299, G=0.587, B=0.114):
    imgstack = np.array(imgstack)
    return imgstack[:, :, :, 0]*R + imgstack[:, :, :, 1]*G + imgstack[:, :, :, 2]*B