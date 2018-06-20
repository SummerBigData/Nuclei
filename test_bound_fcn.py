from boundary_fcn import generator, cvt_to_gray
from keras.models import model_from_json

import sys
from os.path import isfile, join

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

from scipy.misc import imread
from util import *
import matplotlib.pyplot as plt

ids = all_ids()
ids = list(filter(
    lambda id: isfile(join('data', id, 'images', 'bounds_eroded.png')), 
    ids))

paths = [join('data', id, 'images', id+'.png') for id in ids]
X = [imread(path) for path in paths]
X = cvt_to_gray(X)

paths = [join('data', id, 'images', 'bounds_eroded.png') for id in ids]
y = [imread(path) for path in paths]
gen = generator(X, y)

for i in range(5):
    X, y = next(gen)
    pred = model.predict(X)[0, :, :, 0]*255

    _, axs = plt.subplots(1, 3)
    axs[0].imshow(pred, 'gray')
    axs[1].imshow(y[0, :, :, 0], 'gray')
    axs[2].imshow(X[0, :, :, 0], 'gray')
    plt.show()
