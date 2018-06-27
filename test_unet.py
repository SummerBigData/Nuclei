from boundary_fcn import generator
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

gen = generator(X, y)

"""
print 'Calculating mean IoU'
mean_iou, ious = test_model(model, gen, len(ids), ret_ious=True)
plt.hist(ious)
plt.show()
print 'Mean IoU: %f' % mean_iou
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
    axs[0].imshow(pred, 'gray')
    axs[1].imshow(y[0, :, :, 0], 'gray')
    axs[2].imshow(X[0, :, :, 0], 'gray')
    plt.show()
