from keras.models import model_from_json

import cv2
from scipy.misc import imsave

import sys
from os.path import isfile, join

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

from util import *

def generator(X, y, ids):
    while True:
        for mask, bounds, id in zip(X, y, ids):
            mask = mask.reshape(1, mask.shape[0], mask.shape[1], 1)/255.0
            bounds = bounds.reshape(1, bounds.shape[0], bounds.shape[1], 1)/255.0
            yield mask, bounds, id

X, ids = all_recursive_masks(ret_ids=True)
y = masks_for(ids, erode=True)
gen = generator(X, y, ids)

for i in range(len(ids)):
    if i % 10 == 0:
        print '%d / %d' % (i, len(ids))

    X, y, id = next(gen)
    
    pred = X
    for j in range(3):
        pred = model.predict(pred)

    pred = pred[0, :, :, 0]
    act = y[0, :, :, 0]

    _, pred = cv2.threshold(pred, 0.25, 1.0, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3))
    pred = cv2.dilate(pred, kernel, iterations=1) 

    imsave(join('data', id, 'images/sm_rec_mask.png'), pred)
