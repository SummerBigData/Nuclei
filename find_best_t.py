from PIL import Image
import numpy as np
import cv2 as cv
from skimage.morphology import label

import os
from os.path import join

from iou import iou_metric
from util import *

#imgs, ids = all_imgs(ret_ids=True, denoise=True)
#masks = masks_for(ids, erode=True)
img, mask, id = get_img(denoise=True, ret_mask=True, erode=False, ret_id=True)

def num_comps(img):
    return cv.connectedComponentsWithStats(img, cv.CV_32S)[0]-1

import matplotlib.pyplot as plt

# Extremely basic metric for determining whether a given threshold value is "good"
#
# If the number of different pixels in the thresholded image and the mask
# (actually the mean) is less than an arbitrary tolerance, then it is deemed fine.
def is_good_t(thresh, mask):
    if np.sum(thresh == mask) == len(mask.flatten()):
        return 0.0
    return np.mean(thresh != mask)

count = 0

# Try to get the given mask from the input image by first thresholding by t.
#
# Then, go through each quadrant of the image and determine if it is a "good"
# t for that quadrant (see is_good_t).
#
# If it is, then we are done with that quadrant.
#
# Otherwise, recurse on that quadrant of both the image and the mask
def best_t(img, mask, th, orig_shape):
    global count
    count += 1

    th_arr = np.zeros_like(img)
    th_arr += th

    _, res = cv.threshold(img, th, 255, cv.THRESH_BINARY)
    err = is_good_t(res, mask)
    tol = 0.05 * (img.shape[0]*img.shape[1] / float(orig_shape[0]*orig_shape[1]))

    divs = 2
    h = img.shape[0]//divs
    w = img.shape[1]//divs

    for i in range(divs):
        for j in range(divs):
            t, b = i*h, (i+1)*h
            l, r = j*w, (j+1)*w

            img_q = img[t:b, l:r]
            mask_q = mask[t:b, l:r]
            res_q = res[t:b, l:r]

            err = is_good_t(res_q, mask_q)
            #if err < tol or th == 0 or th == 255:
            #    continue
            if t == b or l == r:
                continue

            if np.sum(res_q > 0) > np.sum(mask_q > 0):
                th += 1
            else:
                th -= 1

            res_q, th_arr_q = best_t(img_q, mask_q, th, orig_shape)
            res[t:b, l:r] = res_q
            th_arr[t:b, l:r] = th_arr_q

    return res, th_arr

init_thresh = cv.threshold(img, 25, 255, cv.THRESH_BINARY)[1]
_, axs = plt.subplots(2, 2)

axs[0,0].imshow(init_thresh, 'gray')
axs[0,0].set_title('Thresholded @ 25')
axs[0,0].axis('off')

axs[0,1].imshow(mask, 'gray')
axs[0,1].set_title('Ground Truth')
axs[0,1].axis('off')

res, th_arr = best_t(img, mask, 25, img.shape)

axs[1,0].imshow(res, 'gray')
axs[1,0].set_title('Result - %d Iterations' % count)
axs[1,0].axis('off')

axs[1,1].imshow(th_arr, 'jet')
axs[1,1].set_title('Threshold Map - (%d, %d)' % (th_arr.min(), th_arr.max()))
axs[1,1].axis('off')

print count
plt.show()
