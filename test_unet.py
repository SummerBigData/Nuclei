from unet_white import generator
from keras.models import model_from_json

import sys
from os.path import isfile, join
from util import *

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

from cv2 import dilate
from scipy.misc import imread, imresize
from util import *
import matplotlib.pyplot as plt

X, ids = all_imgs(ret_ids=True, white=True, gray=False)
y = masks_for(ids, erode=False)

s = [512, 256, 128]
for i in range(len(X)):
    for size in s:
        if X[i].shape[0] >= size or X[i].shape[1] >= size:
            new_d = size
            break
    X[i] = imresize(X[i], (size, size, 4))
    y[i] = imresize(y[i], (size, size))
    y[i] = np.expand_dims(y[i], axis=2)

gen = generator(X, y)

#"""
from os.path import join, isfile
filename = join('ious', name+'.txt')
if isfile(filename):
    ious = np.genfromtxt(filename)
    print np.mean(ious)
    plt.hist(ious)
    plt.show()
else:
    print 'Calculating mean IoU'
    mean_iou, ious = test_model(model, gen, len(ids), ret_ious=True)
    np.savetxt(filename, ious)
    plt.hist(ious)
    plt.show()
    print 'Mean IoU: %f' % mean_iou

t = 0.4
low_idxs = np.argwhere(ious <= t).flatten()
hi_idxs = np.argwhere(ious > t).flatten()

print '# Good: %d, # Bad: %d' % (len(hi_idxs), len(low_idxs))
from skimage.morphology import label
from skimage.measure import perimeter

print 'Bad Results'
for _ in range(5):
    idx = np.random.choice(range(len(X)))
    while not idx in low_idxs.tolist():
        idx = np.random.choice(range(len(X)))

    _, ax = plt.subplots(1, 3)
    gray_imshow(ax[0], X[idx], title='Input')
    
    #p = unbatchify(model.predict(batchify(X[idx])))
    x = X[idx]/255.
    p = model.predict(x.reshape(1, x.shape[0], x.shape[1], 4))[0,:,:,0]
    print y[idx].shape, p.shape
    print test_img(y[idx][:,:,0], p)

    gray_imshow(ax[1], np.round(p), title='Predicted')
    gray_imshow(ax[2], y[idx][:,:,0], title='Target')
    plt.show()

print 'Good Results'
for _ in range(5):
    idx = np.random.choice(range(len(X)))
    while not idx in hi_idxs.tolist():
        idx = np.random.choice(range(len(X)))

    _, ax = plt.subplots(1, 3)
    gray_imshow(ax[0], X[idx], title='Input')
    
    #p = unbatchify(model.predict(batchify(X[idx])))
    x = X[idx]/255.
    p = model.predict(x.reshape(1, x.shape[0], x.shape[1], 4))[0,:,:,0]
    print test_img(y[idx][:,:,0], p)

    gray_imshow(ax[1], np.round(p), title='Predicted')
    gray_imshow(ax[2], y[idx][:,:,0], title='Target')
    plt.show()

exit()
#"""

#from find_best_t import plot_best_t

for i in range(5):
    X, y = next(gen)

    pred = model.predict(X)[0, :, :, 0]
    #pred = np.round(pred)
    pred = (pred > 0.5).astype(np.uint8)
    #pred = dilate(pred, np.ones((2, 2)), iterations=1)

    act = y[0, :, :, 0]
    #print np.sqrt(np.sum((pred-act).flatten()**2))
    print test_img(pred, act)

    _, axs = plt.subplots(1, 3)
    gray_imshow(axs[0], pred, title='pred')
    gray_imshow(axs[1], y[0, :, :, 0], title='mask')
    gray_imshow(axs[2], X[0, :, :, 0], title='input')
    plt.show()
