from boundary_fcn import generator
from keras.models import model_from_json
import cv2

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
    lambda id:
        isfile(join('data', id, 'images', 'mask_eroded.png')), ids))

X = all_imgs(ids)
y = masks_for(ids, erode=True) 
gen = generator(X, y)

from find_best_t import best_t, plot_best_t

for i in range(5):
    X, y = next(gen)
    pred = model.predict(X)[0, :, :, 0]

    act = y[0, :, :, 0]
    plot_best_t(img=(pred*255).astype(np.uint8), mask=(act*255).astype(np.uint8)) 
    #print np.sqrt(sum((pred-act).flatten()**2))

    _, axs = plt.subplots(2, 2)
    gray_imshow(axs[0,0], X[0,:,:,0], title='Input')
    gray_imshow(axs[0,1], act, title='Target')
    gray_imshow(axs[1,0], pred, title='Predicted')

    #pred = np.round(pred)
    _, pred = cv2.threshold(pred, 0.25, 1.0, cv2.THRESH_BINARY)

    #res, _ = best_t((pred*255.0).astype(np.uint8), (act*255.).astype(np.uint8), pred.shape)
    gray_imshow(axs[1,1], pred, title='Predicted (thresholded)')
    plt.show()
