from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

from scipy.misc import imread, imresize
from sklearn.feature_extraction.image import extract_patches_2d as extract_patches

import sys
from os import mkdir
from os.path import join, isfile, isdir
from util import *
from iou import *

name = sys.argv[1]
if not isdir(join('models', name)):
    os.mkdir(join('models', name))

from time import time
def patches_generator(inputs, targets, k=8, m=100):
    while True:
        idxs = np.random.choice(range(len(inputs)), len(inputs), replace=False)
        inputs = [inputs[i] for i in idxs]
        targets = [targets[i] for i in idxs]

        # For each image, generate 100 randomly sampled patches from that image
        for x, y in zip(inputs, targets):
            seed = int(time())

            rng = np.random.RandomState(seed)
            x_patches = extract_patches(x/255., (k, k), max_patches=m, random_state=rng)

            rng = np.random.RandomState(seed)
            y_patches = extract_patches(y/255., (k, k), max_patches=m, random_state=rng)
            yield x_patches[:,:,:,np.newaxis], y_patches[:,:,:,np.newaxis]

import tensorflow as tf
from keras import backend as K
K.clear_session()

def mean_iou(y, pred):
    return tf.py_func(batch_iou, [y, pred], tf.float32)

if __name__ == '__main__':
    ids = all_ids()
    ids = list(filter(
        lambda id: isfile(join('data', id, 'images', 'mask_eroded.png')), 
        ids))

    paths = [join('data', id, 'images', id+'.png') for id in ids]
    X = [imread(path) for path in paths]

    import cv2
    X = [cv2.cvtColor(x, cv2.COLOR_BGRA2GRAY) for x in X]

    paths = [join('data', id, 'images', 'mask_eroded.png') for id in ids]
    y = [imread(path) for path in paths]

    inputs = Input((None, None, 1))

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[inputs], outputs=[outputs])

    m = 9*len(X)//10
    X_val, y_val = X[m:], y[m:]
    X, y = X[:m], y[:m]

    patch_size = 64
    batch_size = 25

    gen = patches_generator(X, y, k=patch_size, m=batch_size)
    val_gen = patches_generator(X_val, y_val, k=patch_size, m=batch_size)

    early_stop = EarlyStopping(patience=5, verbose=1)
    checkpoint = ModelCheckpoint(join('models', name, 'model.h5'), 
                                monitor='val_mean_iou', mode='max', 
                                verbose=1, save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[mean_iou])
    with open(join('models', name, 'model.json'), 'w') as f:
        f.write(model.to_json())

    model.fit_generator(
        gen, 
        steps_per_epoch=len(X), 
        epochs=25,
        validation_data=val_gen,
        validation_steps=len(X_val),
        use_multiprocessing=True, workers=4,
        callbacks=[early_stop, checkpoint])

    model.save_weights(join('models', name, 'model.h5'))
