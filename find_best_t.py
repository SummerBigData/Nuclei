from PIL import Image
import numpy as np
import cv2 as cv

import os
from os.path import join

img_ids = os.listdir('data')
img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
img_ids = list(filter(
    lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

X_train, y_train = [], []
for id in img_ids:
    dir = join('data', id, 'images')
    img = np.array(Image.open(join(dir, id+'.png')))
    X_train.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
#    y_train.append(np.array(Image.open(join(dir, 'mask.jpeg'))))
    masks = os.listdir(join('data', id, 'masks'))
    total_mask = np.zeros_like(X_train[-1])
    for mask in masks:
        mask_img = Image.open(join('data', id, 'masks', mask))
        total_mask += np.array(mask_img)
    y_train.append(total_mask)

def num_comps(img):
    return cv.connectedComponentsWithStats(img, cv.CV_32S)[0]-1

import matplotlib.pyplot as plt

def best_t(img, mask):
    t = 25
    _, res = cv.threshold(img, t, 255, cv.THRESH_BINARY)

    _, axs = plt.subplots(1, 2)
    res_ax = axs[0].imshow(res, cmap='gray')
    axs[1].imshow(mask, cmap='gray')

    num_found, num_corr = num_comps(res), num_comps(mask)
    last_diff = 100000
    curr_diff = num_corr - num_found

    #while curr_diff < last_diff:
    last_err = np.inf
    curr_err = np.sqrt(np.sum((res-mask)**2))
    while curr_err < last_err:
        print curr_err, t

        if curr_diff < 0:
            t += 1
        else:
            t -= 1

        _, res = cv.threshold(img, t, 255, cv.THRESH_BINARY)
        res_ax.set_data(res)

        num_found = num_comps(res)
        last_diff = curr_diff
        curr_diff = num_corr - num_found

        last_err = curr_err
        curr_err = np.sqrt(np.sum((res-mask)**2))

    print t, curr_diff, last_diff, num_found, num_corr
    plt.show()

best_t(X_train[0], y_train[0])
best_t(X_train[1], y_train[1])
