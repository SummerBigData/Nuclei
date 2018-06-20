from boundary_fcn import generator
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
    lambda id:
        isfile(join('data', id, 'images', 'mask_eroded.png')), ids))

X = all_imgs(ids)
y = masks_for(ids, erode=True) 
gen = generator(X, y)

for i in range(5):
    X, y = next(gen)
    pred = model.predict(X)[0, :, :, 0]

    act = y[0, :, :, 0]
    #print np.sqrt(sum((pred-act).flatten()**2))

    _, axs = plt.subplots(1, 4)
    axs[0].imshow(X[0,:,:,0], 'gray')
    axs[1].imshow(act, 'gray')
    axs[2].imshow(pred, 'gray')

    #pred = np.round(pred)
    #_, pred = cv2.threshold(pred, 0.33, 1.0, cv2.THRESH_BINARY)
    t = 0.25
    pred[pred > t] = 1.0
    pred[pred <= t] = 0.0

    axs[3].imshow(pred, 'gray')
    plt.show()
