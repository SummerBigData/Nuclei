from util import *
from boundary_fcn import generator
from keras.models import model_from_json

import sys
import os
from os.path import isfile, join

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

import cv2 as cv
from scipy.misc import imread, imresize
from util import *
import matplotlib.pyplot as plt

ids = os.listdir('test_data2')
ids = np.random.choice(ids, 200)
X = [imread(join('test_data2', id, 'images', id+'.png')) for id in ids]
X = [cv.cvtColor(x, cv.COLOR_BGRA2GRAY) for x in X if len(x.shape) == 3]

s = [1024, 512, 256, 128]
for i in range(len(X)):
    for size in s:
        if X[i].shape[0] >= size or X[i].shape[1] >= size:
            new_shape = (size, size)
            break
    X[i] = imresize(X[i], new_shape)

for _ in range(5):
    x = X[np.random.randint(0, len(X))]
    while np.mean(x) >= 127.5:
        x = X[np.random.randint(0, len(X))]
    x = batchify(x/255.)

    pred = model.predict(x)[0, :, :, 0]
    pred = np.round(pred)

    _, axs = plt.subplots(1, 2)
    axs[1].imshow(pred, 'gray')
    axs[0].imshow(unbatchify(x), 'gray')
    plt.show()
