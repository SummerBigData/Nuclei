from unet import generator

import numpy as np
import pandas as pd

from scipy.misc import imread, imresize
from skimage.morphology import label
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY as bgr2gray

import os
from os.path import join

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

# Create the actual submission for a given FCN
def generate_sub(name_black, name, name_white=None):
    print 'Creating submission for %s' % name

    # For some reason if the models are created in a different script and
    # then passed as parameters to this function, then Tensorflow breaks.
    #
    # Therefore, just pass the model names and load them from json in this function.
    from keras.models import model_from_json

    with open(join('models', name_black, 'model.json')) as f:
        json = f.read()

    model_black = model_from_json(json)
    model_black.load_weights(join('models', name_black, 'model.h5'))

    if not name_white is None:
        with open(join('models', name_white, 'model.json')) as f:
            json = f.read()

        model_white = model_from_json(json)
        model_white.load_weights(join('models', name_white, 'model.h5'))

    def load_imgs(d):
        ids = os.listdir(d)
        imgs = [imread(join(d, id, 'images', id+'.png')) for id in ids]

        # Some test images are already 2d (grayscale) so don't convert them
        for i, x in enumerate(imgs):
            if len(x.shape) == 3:
                imgs[i] = cvtColor(x, bgr2gray)

        # Keep the sizes so that after passing through the unet, they can be
        # reshaped into their original size
        shapes = [x.shape for x in imgs]

        return imgs, ids, shapes

    X, ids, sizes = load_imgs('test_data1')
    #X_tmp, ids_tmp, size_tmp = load_imgs('test_data1')
    #X.extend(X_tmp)
    #sizes.extend(size_tmp)
    #ids.extend(ids_tmp)

    # Reshape each image to either 512, 256, or 128, whichever is closest.
    #
    # Do this because the unet uses concatenate layers and if a lot of sized
    # images are passed through the convolution layers, their dimension changes
    # and this breaks Keras.
    s = [512, 256, 128]
    for i in range(len(X)):
        for size in s:
            if X[i].shape[0] >= size or X[i].shape[1] >= size:
                new_shape = (size, size)
                break
        X[i] = imresize(X[i], new_shape)


    rle_ids = []
    rles = []
    for i, x in enumerate(X):
        if i % 100 == 0:
            print '%d / %d' % (i, len(X))
       
        batch = x.reshape(1, x.shape[0], x.shape[1], 1)
        if not model_white is None and np.mean(x) >= 127.5:
            p = model_white.predict(batch)[0,:,:,0]
        else:
            p = model_black.predict(batch)[0,:,:,0]
        p = imresize(p, sizes[i])

        """
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(imresize(x, sizes[i]), 'gray')
        axs[1].imshow(p, 'gray')
        plt.show()
        """

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

    sub = pd.DataFrame()
    sub['ImageId'] = rle_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(join('subs', name+'.csv'), index=False)
