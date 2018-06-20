from boundary_fcn import generator
from keras.models import model_from_json

import sys
from os.path import isfile, join

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

from scipy.misc import imread, imresize
from util import *
import matplotlib.pyplot as plt

ids = all_ids()
ids = list(filter(
    lambda id:
        isfile(join('data', id, 'images', 'mask_eroded.png')), ids))

paths = [join('data', id, 'images', id+'.png') for id in ids]
X = [imread(path) for path in paths]

import cv2
X = [cv2.cvtColor(x, cv2.COLOR_BGRA2GRAY) for x in X]

paths = [join('data', id, 'images', 'mask_eroded.png') for id in ids]
y = [imread(path) for path in paths]

s = [512, 256, 128]
for i in range(len(X)):
    for size in s:
        if X[i].shape[0] >= size or X[i].shape[1] >= size:
            new_shape = (size, size)
            break
    X[i] = imresize(X[i], new_shape)
    y[i] = imresize(y[i], new_shape)

gen = generator(X, y)

for i in range(5):
    X, y = next(gen)
    pred = model.predict(X)[0, :, :, 0]
    #pred = np.round(pred)
    pred = (pred > 0.5).astype(np.uint8)

    act = y[0, :, :, 0]
    print np.sqrt(np.sum((pred-act).flatten()**2))

    _, axs = plt.subplots(1, 3)
    axs[0].imshow(pred, 'gray')
    axs[1].imshow(y[0, :, :, 0], 'gray')
    axs[2].imshow(X[0, :, :, 0], 'gray')
    plt.show()
