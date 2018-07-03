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
        
import tensorflow as tf
from keras import backend as K
K.clear_session()

def mean_iou(y, pred):
    return tf.py_func(batch_iou, [y, pred], tf.float32)

from keras.losses import binary_crossentropy as bce
from skimage.measure import perimeter

def batch_loss_perim(y_batch, p_batch):
    print 'blp'
    m = float(y_batch.shape[0])
    bce_loss = sum(bce(y, p) for (y, p) in zip(y_batch, p_batch))/m
    perim_loss = sum(perimeter(np.round(p))/np.count_nonzero(np.round(p)) for p in p_batch)/m
    return bce_loss + perim_loss 

def loss_perim(y, p):
    print 'lp'
    print y.shape, p.shape
    print K.is_placeholder(y), K.is_placeholder(p)
    if not K.is_placeholder(y):
        print 'seflikjsdf'
        return tf.py_func(batch_loss_perim, [y, p], tf.float32)
    return K.constant(np.inf)

if __name__ == '__main__':
    X, ids = all_imgs(ret_ids=True, white=False)
    y = masks_for(ids, erode=False)

    X = arr([imresize(x, (256, 256))/255. for x in X])
    y = arr([imresize(tr, (256, 256))/255. for tr in y])

    X = np.expand_dims(X, axis=3)
    y = np.expand_dims(y, axis=3)

    from unet_arch import unet_model
    model = unet_model(dropout=0.0, inpt_shape=(256, 256, 1))

    num_aug = 1
    m = 9*len(X)//10
    X_val, y_val = X[m:], y[m:]
    X, y = X[:m], y[:m]

    early_stop = EarlyStopping(patience=7, verbose=1)
    checkpoint = ModelCheckpoint(join('models', name, 'model.h5'), 
                                monitor='val_mean_iou', mode='max', 
                                verbose=1, save_best_only=True)

    model.compile(loss=loss_perim, optimizer='rmsprop', metrics=[mean_iou])
    with open(join('models', name, 'model.json'), 'w') as f:
        f.write(model.to_json())

    model.fit(X, y, batch_size=64, epochs=35,
            validation_data=(X_val, y_val), callbacks=[early_stop, checkpoint])
    model.save_weights(join('models', name, 'model.h5'))
