from util import *

from keras.callbacks import EarlyStopping, ModelCheckpoint
from unet_arch import unet_model

from scipy.misc import imread, imresize

import sys
from os import mkdir
from os.path import join, isfile, isdir
from iou import *
from time import time

def augment(inpts, targets, num_aug=5):
    X_set, y_set = [], []
    for inpt, target in zip(inpts, targets):
        for _ in range(num_aug):
            x, y = apply_transform(inpt, target)
            X_set.append(np.expand_dims(x, axis=2))
            y_set.append(np.expand_dims(y, axis=2))

    return X_set, y_set
        

def generator(X, y):
    while True:
        idxs = np.random.choice(range(len(X)), len(X), replace=False)
        X = [X[i] for i in idxs]
        y = [y[i] for i in idxs]

        for inpt, target in zip(X, y):
            inp, targ = np.expand_dims(inpt, axis=0), np.expand_dims(target, axis=0)
            yield inp, targ

import tensorflow as tf
from keras import backend as K
K.clear_session()

def mean_iou(y, pred):
    return tf.py_func(batch_iou, [y, pred], tf.float32)

if __name__ == '__main__':
    name = sys.argv[1]
    if not isdir(join('models', name)):
        os.mkdir(join('models', name))

    X, ids = all_imgs(white=True, ret_ids=True, gray=False)
    y = masks_for(ids, erode=False)

    s = [512, 256, 128]
    for i in range(len(X)):
        for size in s:
            if X[i].shape[0] >= size or X[i].shape[1] >= size:
                new_d = size
                break

        X[i] = imresize(X[i]/255., (new_d, new_d, 4))
        y[i] = imresize(y[i]//255, (new_d, new_d))
        y[i] = np.expand_dims(y[i], axis=2)

    m = 9*len(X)//10
    X_val, y_val = X[m:], y[m:]
    X, y = X[:m], y[:m]

    '''
    X_val, y_val = get_test_data()
    for i in range(len(X_val)):
        for size in s:
            if X_val[i].shape[0] >= size or X_val[i].shape[1] >= size:
                new_d
                break

        X_val[i] = imresize(X_val[i]/255., (new_d, new_d, 4))
        y_val[i] = imresize(y_val[i]//255, (new_d, new_d))
        y_val[i] = np.expand_dims(y_val[i], axis=2)
    '''

    num_aug = 1
    #gen = generator(*augment(X, y, num_aug=num_aug))
    gen = generator(X, y)
    val_gen = generator(X_val, y_val)

    early_stop = EarlyStopping(patience=7, verbose=1)
    checkpoint = ModelCheckpoint(join('models', name, 'model.h5'), 
                                monitor='val_mean_iou', mode='max', 
                                verbose=1, save_best_only=True)

    model = unet_model(inpt_shape=(None, None, 4))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[mean_iou])
    with open(join('models', name, 'model.json'), 'w') as f:
        f.write(model.to_json())

    model.fit_generator(
        gen, 
        steps_per_epoch=len(X)*num_aug, 
        epochs=35,
        validation_data=val_gen,
        validation_steps=len(X_val),
        use_multiprocessing=True, workers=4,
        callbacks=[early_stop, checkpoint])
    model.save_weights(join('models', name, 'model.h5'))
