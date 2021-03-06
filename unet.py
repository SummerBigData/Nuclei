from util import *
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.preprocessing.image import apply_transform, random_transform
from keras.preprocessing.image import ImageDataGenerator

from scipy.misc import imread, imresize

import sys
from os import mkdir
from os.path import join, isfile, isdir
from iou import *
from time import time

name = sys.argv[1]
if not isdir(join('models', name)):
    os.mkdir(join('models', name))

def augment(inpts, targets, num_aug=5):
    datagen = ImageDataGenerator()
    X_set, y_set = [], []
    for inpt, target in zip(inpts, targets):
        inpt = np.expand_dims(inpt/255., axis=2)
        target = np.expand_dims(target/255., axis=2)
        for _ in range(num_aug):
            seed = int(time())
            x = datagen.random_transform(inpt, seed=seed)
            y = datagen.random_transform(target, seed=seed)
            X_set.append(x)
            y_set.append(y)
    return X_set, y_set
        

def generator(X, y):
    while True:
        idxs = np.random.choice(range(len(X)), len(X), replace=False)
        X = [X[i] for i in idxs]
        y = [y[i] for i in idxs]

        for inpt, target in zip(X, y):
            yield np.expand_dims(inpt, axis=0), np.expand_dims(target, axis=0)

import tensorflow as tf
from keras import backend as K
K.clear_session()

def mean_iou(y, pred):
    return tf.py_func(batch_iou, [y, pred], tf.float32)

if __name__ == '__main__':
    X, ids = all_imgs(ret_ids=True, white=True)
    y = masks_for(ids, erode=False)

    s = [512, 256, 128]
    for i in range(len(X)):
        for size in s:
            if X[i].shape[0] >= size or X[i].shape[1] >= size:
                new_shape = (size, size)
                break

        X[i] = imresize(X[i], new_shape)
        y[i] = imresize(y[i], new_shape)

    from unet_arch import unet_model
    model = unet_model(dropout=0.0)

    num_aug = 3
    m = 9*len(X)//10
    X_val, y_val = X[m:], y[m:]
    X, y = X[:m], y[:m]
    gen = generator(*augment(X, y, num_aug=num_aug))
    val_gen = generator(*augment(X_val, y_val, num_aug=num_aug))
    #gen = generator(X, y)
    #val_gen = generator(X_val, y_val)

    early_stop = EarlyStopping(patience=7, verbose=1)
    checkpoint = ModelCheckpoint(join('models', name, 'model.h5'), 
                                monitor='val_mean_iou', mode='max', 
                                verbose=1, save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[mean_iou])
    with open(join('models', name, 'model.json'), 'w') as f:
        f.write(model.to_json())

    model.fit_generator(
        gen, 
        steps_per_epoch=len(X)*num_aug, 
        epochs=35,
        validation_data=val_gen,
        validation_steps=len(X_val)*num_aug,
        use_multiprocessing=True, workers=4,
        callbacks=[early_stop, checkpoint])
    model.save_weights(join('models', name, 'model.h5'))
