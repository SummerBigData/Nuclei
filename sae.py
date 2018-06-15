from keras.models import Sequential
from keras.layers import Input, Dense

from PIL import Image
import numpy as np
import cv2 as cv

import os
from os.path import join

img_ids = os.listdir('data')
img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
img_ids = list(filter(
    lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

X_train, y_train = [], []
for id in img_ids:
    dir = join('data', id, 'images')
    img = np.array(Image.open(join(dir, id+'.png')))
    if not img.shape == (256, 256, 4):
        continue
    X_train.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY)/255.0)
    y_train.append(np.array(Image.open(join(dir, 'mask.jpeg')))/255.0)

model = Sequential([
    Dense(100000, input_shape=(256*256,), activation='sigmoid'),
    Dense(256*256, activation='sigmoid')])
model.compile(optimizer='rmsprop', loss='mse')

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0], -1)
model.fit(X_train, y_train, batch_size=64, epochs=5)
