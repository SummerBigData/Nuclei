from boundary_fcn import generator
from keras.models import model_from_json

import cv2
from scipy.misc import imsave

import sys
import subprocess

import os
from os.path import isfile, join

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

from util import *
import matplotlib.pyplot as plt

X, ids = all_recursive_masks(ret_ids=True)
y = masks_for(ids, erode=True)
gen = generator(X, y)

for i in range(5):
    X, y = next(gen)
    
    pred = X

    # Uncomment the following blocks if you want to create a gif
    # of the smoothing process.
    """
    files = ['smoothed_0.png']
    imsave(join('images', files[-1]), pred[0,:,:,0])
    """
    
    for j in range(3):
        pred = model.predict(pred)
        """
        files.append('smoothed_%d.png' % (j+1))
        imsave(join('images', files[-1]), pred[0,:,:,0])
        """

    """
    files = [join('images', f) for f in files]
    args = ['convert', '-delay', '50', '-loop', '0']
    args.extend(files)
    args.append('images/smoothing_%d.gif' % i)
    subprocess.call(args)

    for f in files:
        os.remove(f)
    """

    pred = pred[0, :, :, 0]
    act = y[0, :, :, 0]

    _, axs = plt.subplots(2, 2)
    gray_imshow(axs[0,0], X[0,:,:,0], title='Original (unsmoothed)')
    gray_imshow(axs[0,1], act, title='Target mask')
    gray_imshow(axs[1,0], pred, title='Predicted mask')

    #pred = np.round(pred)
    _, pred = cv2.threshold(pred, 0.25, 1.0, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3))
    pred = cv2.dilate(pred, kernel, iterations=1) 

    gray_imshow(axs[1,1], pred, title='Predicted (threshold+dilate)')
    plt.show()
