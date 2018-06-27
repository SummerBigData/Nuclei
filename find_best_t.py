import numpy as np
import cv2 as cv
from skimage.morphology import label
from scipy.misc import imsave

import os
from os.path import join

from iou import iou_metric
from util import *

import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2

with open('models/smoothing/model.json') as f:
    json = f.read()
model = model_from_json(json)
model.load_weights('models/smoothing/model.h5')


""" Goodness metric """
# Extremely basic metric for determining whether a given threshold value is "good"
#
# If the number of different pixels in the thresholded image and the mask
# (actually the mean) is less than an arbitrary tolerance, then it is deemed fine.
def threshold_err(thresh, mask):
    if np.sum(thresh == mask) == len(mask.flatten()):
        return 0.0
    return np.mean(thresh != mask)


# Global variable to keep track of how many times best_t is recursed upon
count = 0

""" Best threshold computation """
# Try to get the given mask from the input image by first thresholding by t.
#
# Then, go through each quadrant of the image and determine if it is a "good"
# t for that quadrant (see threshold_err).
#
# If it is, then we are done with that quadrant.
#
# Otherwise, recurse on that quadrant of both the image and the mask
def best_t(img, mask, orig_shape, ttype=cv.THRESH_BINARY, prev_th=0):
    global count
    count += 1

    # Create an array of the threshold value for display in the heatmap
    #th = int(np.mean(img))

    if np.count_nonzero(img) == 0:
        th_arr = np.zeros_like(img, dtype=np.uint16)
        th_arr += prev_th
        return img, th_arr
    th = int(np.mean(img[img>0]))
    th_arr = np.zeros_like(img, dtype=np.uint16)
    th_arr += th

    # Threshold the input image according to the given threshold
    _, res = cv.threshold(img, th, 255, ttype)

    # Scale the original tolerance of 0.035 by 2 each time the image is recursed upon.
    tol = 0.025 * (img.shape[0]*img.shape[1] / float(orig_shape[0]*orig_shape[1]))

    # When divs = 2, then the given image and mask are split up into
    # quadrants. If divs = 3, then it is split up into ninths and so on.
    divs = 2
    h = img.shape[0]//divs
    w = img.shape[1]//divs

    for i in range(divs):
        for j in range(divs):
            # Calculate the top, bottom, left, and right of the quadrant
            t, b = i*h, (i+1)*h
            l, r = j*w, (j+1)*w

            # If any dimension of the quadrant is 0, then we are down
            # to the pixel level and there's no more improving to do.
            if t == b or l == r:
                continue

            # Obtain the quadrants from the image, mask, and thresholded image
            img_q = img[t:b, l:r]
            mask_q = mask[t:b, l:r]
            res_q = res[t:b, l:r]

            # Calculate the error between the thresholded image and the truth mask
            # If it is below the tolerance (size-dependent), then stop recursing
            # and say that the threshold is good enough for that region.
            err = threshold_err(res_q, mask_q)
            if err < tol:
                continue

            # If there are more white pixels in the thresholded image, assume that
            # the threshold value needs to be increased as to exlude more pixels
            #
            # Otherwise, lower the threshold value to include more pixels
            if np.sum(res_q > 0) > np.sum(mask_q > 0):
                th += 1
            else:
                th -= 1

            res_q, th_arr_q = best_t(img_q, mask_q, orig_shape, ttype=ttype)
            res[t:b, l:r] = res_q
            th_arr[t:b, l:r] = th_arr_q

    return res, th_arr


""" Plotting """
def plot_best_t(img=None, mask=None, th_arr=None, res=None):
    if img is None or mask is None:
        img, mask, id = get_img(denoise=False, ret_mask=True, erode=True, ret_id=True)

    if np.mean(img) > 127.5:
        img = 255-img

    tval = int(np.mean(img[img>0]))
    init_thresh = cv.threshold(img, tval, 255, cv.THRESH_BINARY)[1]
    _, axs = plt.subplots(2, 3)

    gray_imshow(axs[0,0], img, title='Input')
    gray_imshow(axs[1,0], init_thresh, title='Thresholded @ %d' % tval)
    gray_imshow(axs[0,1], mask, title='Ground Truth (Eroded)')

    if res is None:
        ttype = cv.THRESH_BINARY
        #if np.mean(img) > 127.0:
        #    ttype = cv.THRESH_BINARY_INV
        res, th_arr = best_t(img, mask, img.shape, ttype=ttype)

    gray_imshow(axs[0,2], th_arr, cmap='jet', title=\
        'Threshold Map - (%d, %d)' % (th_arr.min(), th_arr.max()))
    #gray_imshow(axs[0,2], mask_for(id, erode=False), title='Ground Truth (Original)')
    gray_imshow(axs[1,1], res, title='Result - %d Iterations' % count)


    # Propagate the thresholded image through the pretrained smoothing
    # FCN to try and smooth it.
    res = res.reshape(1, res.shape[0], res.shape[1], 1)
    for _ in range(2):
        res = model.predict(res)
    res = res[0,:,:,0]

    # Threshold the smoothed image and dilate it to achieve best results.
    _, res = cv2.threshold(res, 0.25, 1.0, cv2.THRESH_BINARY)
    #kernel = np.ones((3, 3))
    #res = cv2.dilate(res, kernel, iterations=1) 

    print 'IoU: %f' % test_img(res, mask)

    gray_imshow(axs[1,2], res, title='Result (smoothed)')
    plt.tight_layout(w_pad=0.0, h_pad=0.0)
    plt.show()


def save_best_t():
    imgs, ids = all_imgs(denoise=True, ret_ids=True)
    masks = masks_for(ids, erode=True)    
    
    for i, (id, img, mask) in enumerate(zip(ids, imgs, masks)):
        ttype = cv.THRESH_BINARY
        if np.mean(img) > 127.0:
            ttype = cv.THRESH_BINARY_INV

        res, _ = best_t(img, mask, img.shape, ttype=ttype)
        imsave(join('data', id, 'images', 'thresh_recursive.png'), res)

if __name__ == '__main__':
    plot_best_t()
