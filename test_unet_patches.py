from boundary_fcn import generator
from unet_patches import patches_generator
from keras.models import model_from_json

import sys
from os.path import isfile, join
from util import *

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
    json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

from scipy.misc import imread, imresize
from util import *
import matplotlib.pyplot as plt

X, ids = all_imgs(ret_ids=True)
y = masks_for(ids, erode=True)
gen = patches_generator(X, y, k=64, m=25)

"""
print 'Calculating mean IoU'
mean_iou, ious = test_model(model, gen, len(ids), ret_ious=True, patches=True)
plt.hist(ious)
plt.show()
print 'Mean IoU: %f' % mean_iou
"""

#from find_best_t import plot_best_t
gen = generator(X, y)

for i in range(5):
    X, y = next(gen)
    if X.shape[1] % 64 == 0 and X.shape[2] % 64 == 0:
        continue

    pred = np.zeros_like(X[0,:,:,0])
    for i in range(X.shape[1]//64):
        for j in range(X.shape[2]//64):
            patch = X[0, i*64 : (i+1)*64, j*64 : (j+1)*64, 0]
            patch = patch.reshape(1, patch.shape[0], patch.shape[1], 1)
            pred_patch = model.predict(patch)[0, :, :, 0]
            pred[i*64 : (i+1)*64, j*64 : (j+1)*64] = pred_patch

    pred = np.round(pred)
    act = y[0, :, :, 0]
    print test_img(pred, act)

    _, axs = plt.subplots(1, 3)
    axs[0].imshow(pred, 'gray')
    axs[1].imshow(y[0, :, :, 0], 'gray')
    axs[2].imshow(X[0, :, :, 0], 'gray')
    plt.show()
