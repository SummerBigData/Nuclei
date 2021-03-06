from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.optimizers import SGD

from scipy.misc import imread

import sys
from os import mkdir
from os.path import join, isfile, isdir
from util import *

name = sys.argv[1]
if not isdir(join('models', name)):
    os.mkdir(join('models', name))

def generator(X, y):
    while True:
        idxs = np.random.choice(range(len(X)), len(X), replace=False)
        X = [X[i] for i in idxs]
        y = [y[i] for i in idxs]

        for mask, bounds in zip(X, y):
            mask = mask.reshape(1, mask.shape[0], mask.shape[1], 1)/255.0
            bounds = bounds.reshape(1, bounds.shape[0], bounds.shape[1], 1)/255.0
            yield mask, bounds

if __name__ == '__main__':
    ids = all_ids()
    ids = list(filter(
        lambda id:
            isfile(join('data', id, 'images', id+'.png')) and
            isfile(join('data', id, 'images', 'mask_eroded.png')), ids))

    print 'Loading inputs and denoising'
    X = all_imgs(ids=ids, denoise=True)

    print 'Loading masks (eroded)'
    y = masks_for(ids, erode=True)
    gen = generator(X, y)

    w = 2
    model = Sequential([
        Conv2D(8, (w, w), 
            padding='same', 
            activation='relu', 
            input_shape=(None, None, 1)),
        Conv2DTranspose(1, (w, w), 
            padding='same', 
            activation='sigmoid')
    ])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.fit_generator(gen, steps_per_epoch=len(ids), epochs=4)

    with open(join('models', name, 'model.json'), 'w') as f:
        f.write(model.to_json())
    model.save_weights(join('models', name, 'model.h5'))
