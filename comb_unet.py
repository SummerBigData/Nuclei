from boundary_fcn import generator, cvt_to_gray
from keras.models import model_from_json

import sys
from os.path import isfile, join
from util import *

from scipy.misc import imread, imresize
from util import *
import matplotlib.pyplot as plt

name_black = sys.argv[1]
name_white = sys.argv[2]

"""
from create_sub import generate_sub
generate_sub(name_black, 'comb-unet-s1', name_white=name_white)
exit()
#"""

with open(join('models', name_black, 'model.json')) as f:
    json = f.read()

model_black = model_from_json(json)
model_black.load_weights(join('models', name_black, 'model.h5'))

with open(join('models', name_white, 'model.json')) as f:
    json = f.read()

model_white = model_from_json(json)
model_white.load_weights(join('models', name_white, 'model.h5'))

#X, ids = all_imgs(ret_ids=True, white=None)
#y = masks_for(ids, erode=False)
X, y = get_test_data()

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
mean_iou, ious = test_model(model_black, gen, len(ids), ret_ious=True, model_white=model_white)
print 'Mean IoU: %f' % mean_iou
plt.hist(ious)
plt.show()
exit()
#"""

'''
for i in range(5):
    X, y = next(gen)
    #while np.mean(X[0,:,:,0]) < 127.5:
    #    X, y = next(gen)

    if np.mean(X[0,:,:,0]) >= X[0,:,:,0].max()/2.:
        pred = model_white.predict(X)[0,:,:,0]
    else:
        pred = model_black.predict(X)[0,:,:,0]

    pred = np.round(pred)
    act = y[0, :, :, 0]
    print test_img(pred, act)

    _, axs = plt.subplots(1, 3)
    gray_imshow(axs[0], X[0,:,:,0], title='Input')
    gray_imshow(axs[2], y[0,:,:,0], title='Target')
    gray_imshow(axs[1], pred, title='Predicted')
    plt.show()
'''

def show(inpt, pred, target, iou):
    _, ax = plt.subplots(1, 3)
    gray_imshow(ax[0], inpt, title='Input')
    gray_imshow(ax[1], pred, title='Pred, IoU: %f' % iou)
    gray_imshow(ax[2], target, title='Ground Truth')
    plt.show()

iou_b = []
iou_w = []
num_w = 0
num_b = 0
for _ in range(len(X)):
    X, y = next(gen)
    
    if np.mean(X[0,:,:,0]) >= X[0,:,:,0].max()/2.:
        num_w += 1
        p = model_white.predict(X)[0,:,:,0]
        iou_w.append(test_img(p, y[0,:,:,0]))
        show(X[0,:,:,0], np.round(p), y[0,:,:,0], iou_w[-1])
    else:
        num_b += 1
        p = model_black.predict(X)[0,:,:,0]
        #iou_b.append(test_img(p, y[0,:,:,0]))
print num_w, num_b
exit()

print 'Mean iou (b): %f' % arr(iou_b).mean()
plt.hist(iou_b)
plt.title('IoU Black')
plt.show()

print 'Mean iou (w): %f' % arr(iou_w).mean()
plt.hist(iou_w)
plt.title('IoU White')
plt.show()

ious = iou_b+iou_w
print 'Mean iou (tot): %f' % arr(ious).mean()
plt.hist(ious)
plt.title('IoU Total')
plt.show()
