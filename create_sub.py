from unet import generator
from keras.models import model_from_json

import numpy as np

import sys
import os
from os.path import join

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

from scipy.misc import imread, imresize
from skimage.morphology import label
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY as bgr2gray

ids = os.listdir('test_data2')
X = [imread(join('test_data2', id, 'images', id+'.png')) for id in ids]
for i, x in enumerate(X):
    if len(x.shape) == 3:
        X[i] = cvtColor(x, bgr2gray)
sizes = [x.shape for x in X]

s = [512, 256, 128]
for i in range(len(X)):
    for size in s:
        if X[i].shape[0] >= size or X[i].shape[1] >= size:
            new_shape = (size, size)
            break
    X[i] = imresize(X[i], new_shape)

# Perform run-length encoding on a given mask
# I stole this code from Kaggle and have not put in the effort
# to learn how it works.
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# Separate each connected component from the total mask image
# and perform rle on it.
def labels_to_rles(x):
    for i in range(1, x.max()+1):
        yield rle_encoding(x == i)

rle_ids = []
rles = []
for i, x in enumerate(X):
    if i % 100 == 0:
        print '%d / %d' % (i, len(X))
   
    p = model.predict(x.reshape(1, x.shape[0], x.shape[1], 1))[0,:,:,0]
    p = imresize(p, sizes[i])

    import matplotlib.pyplot as plt
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(imresize(x, sizes[i]), 'gray')
    axs[1].imshow(p, 'gray')
    plt.show()

    labels = label(p > 0.5)
    x_rles = list(labels_to_rles(labels))

    """
    if len(x_rles) == 0:
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(imresize(x, sizes[i]), 'gray')
        axs[1].imshow(p, 'gray')
        plt.show()
        exit()
    """

    rles.extend(x_rles)
    rle_ids.extend([ids[i]] * len(x_rles))

import pandas as pd
sub = pd.DataFrame()
sub['ImageId'] = rle_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(join('models', name, 'sub.csv'), index=False)
