from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from unet_arch import unet_model

from scipy.misc import imread, imresize

import sys
from os import mkdir
from os.path import join, isfile, isdir
from util import *
from iou import *

name = sys.argv[1]
if not isdir(join('models', name)):
    os.mkdir(join('models', name))

def generator(X, y):
    while True:
        idxs = np.random.choice(range(len(X)), len(X), replace=False)
        X = [X[i] for i in idxs]
        y = [y[i] for i in idxs]

        for inpt, targ in zip(X, y):
            inpt = inpt.reshape(1, inpt.shape[0], inpt.shape[1], 1)/255.0
            targ = targ.reshape(1, targ.shape[0], targ.shape[1], 1)/255.0
            yield inpt, targ

import tensorflow as tf
from keras import backend as K
K.clear_session()

def mean_iou(y, pred):
    return tf.py_func(batch_iou, [y, pred], tf.float32)

if __name__ == '__main__':
    X, ids = all_imgs(white=True, ret_ids=True)
    y = masks_for(ids, erode=False)

    s = [512, 256, 128]
    for i in range(len(X)):
        for size in s:
            if X[i].shape[0] >= size or X[i].shape[1] >= size:
                new_shape = (size, size)
                break

        X[i] = imresize(X[i], new_shape)
        y[i] = imresize(y[i], new_shape)

    # ZCA whiten all of the images
    """
    datagen = ImageDataGenerator(zca_whitening=True)
    for i, x in enumerate(X):
        batch = x.reshape(1, x.shape[0], x.shape[1], 1)
        datagen.fit(batch)
        X[i] = datagen.flow(batch, batch_size=1)
    #"""

    m = 9*len(X)//10
    X_val, y_val = X[m:], y[m:]
    X, y = X[:m], y[:m]
    gen = generator(X, y)
    val_gen = generator(X_val, y_val)

    early_stop = EarlyStopping(patience=7, verbose=1)
    checkpoint = ModelCheckpoint(join('models', name, 'model.h5'), 
                                monitor='val_mean_iou', mode='max', 
                                verbose=1, save_best_only=True)

    model = unet_model()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[mean_iou])
    with open(join('models', name, 'model.json'), 'w') as f:
        f.write(model.to_json())

    model.fit_generator(
        gen, 
        steps_per_epoch=len(X), 
        epochs=35,
        validation_data=val_gen,
        validation_steps=len(X_val),
        use_multiprocessing=True, workers=4,
        callbacks=[early_stop, checkpoint])
    model.save_weights(join('models', name, 'model.h5'))
