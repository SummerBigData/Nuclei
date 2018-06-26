from boundary_fcn import generator
from keras.models import model_from_json

import sys
from os.path import isfile, join
from util import *

whole_name = sys.argv[1]
patch_name = sys.argv[2]

with open(join('models', whole_name, 'model.json')) as f:
    json = f.read()

model_whole = model_from_json(json)
model_whole.load_weights(join('models', whole_name, 'model.h5'))

with open(join('models', patch_name, 'model.json')) as f:
    json = f.read()

model_patch = model_from_json(json)
model_patch.load_weights(join('models', patch_name, 'model.h5'))

from scipy.misc import imread, imresize
from util import *
import matplotlib.pyplot as plt

X, ids = all_imgs(ret_ids=True)
y = masks_for(ids, erode=True)

s = [512, 256, 128]
for i in range(len(X)):
    for size in s:
        if X[i].shape[0] >= size or X[i].shape[1] >= size:
            new_shape = (size, size)
            break
    X[i] = imresize(X[i], new_shape)
    y[i] = imresize(y[i], new_shape)

gen = generator(X, y)

"""
print 'Calculating mean IoU'
mean_iou, ious = test_model(model, gen, len(ids), ret_ious=True)
plt.hist(ious)
plt.show()
print 'Mean IoU: %f' % mean_iou
"""

#from find_best_t import plot_best_t

for i in range(5):
    X, y = next(gen)

    pred_whole = np.round(model_whole.predict(X)[0, :, :, 0])
    pred_patch = np.round(model_patch.predict(X)[0, :, :, 0])

    act = y[0, :, :, 0]
    print 'Whole IoU: %f' % test_img(pred_whole, act)
    print 'Patch IoU: %f' % test_img(pred_patch, act)
    print '\n***\n'

    _, axs = plt.subplots(2, 2)
    gray_imshow(axs[0,0], X[0,:,:,0], title='Input')
    gray_imshow(axs[0,1], y[0,:,:,0], title='Target')
    gray_imshow(axs[1,0], pred_whole, title='Predicted (whole)')
    gray_imshow(axs[1,1], pred_patch, title='Predicted (patches)')
    plt.show()
