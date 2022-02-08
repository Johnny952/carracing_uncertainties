import json
import numpy as np

def string2image(data):
    dtype = data['dtype']
    shape = data['shape']
    img = np.fromstring(data['img'].encode('latin-1'), dtype=dtype).reshape(shape)
    return img
