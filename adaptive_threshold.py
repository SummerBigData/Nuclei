from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

import os
from os.path import join
from util import *

img_arr, total_mask, id = get_img(denoise=False, ret_mask=True, erode=True, ret_id=True)
num_mask = len(os.listdir(join('data', id, 'masks')))

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2)

def conn_comp(img):
    output = cv.connectedComponentsWithStats(img, cv.CV_32S)
    return output[0]-1, output[1]


num_labels, labels = conn_comp(total_mask.astype(np.uint8))
print 'Number of masks: %d' % num_mask
print 'Number of components in original: %d' % num_labels
num_labels_true = num_labels

axs[0,0].imshow(total_mask, cmap='gray')
axs[0,1].imshow(labels, cmap='jet')


# Use Gaussian adaptive thresholding to try and find the perfect mask from the input
# image. Doesn't work that well (at least with current parameters).
img_thresh = cv.adaptiveThreshold(img_arr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
num_labels, thresh_labels = conn_comp(img_thresh)


# For each component that isn't the background, remove it if it is small.
# This is done because the adaptive thresholding currently finds a lot of tiny,
# individual components.
#
# Small is determined to be less than 8 pixels.
for i in range(1, num_labels+1):
    if np.sum(thresh_labels == i) <= 7:
        num_labels -= 1
        thresh_labels[thresh_labels == i] = 0
img_thresh = np.zeros_like(img_thresh)
img_thresh[thresh_labels > 0] = 255


axs[1,0].imshow(img_thresh, cmap='gray')
axs[1,1].imshow(thresh_labels, cmap='jet')
print 'Number of components in thresholded (Gaussian): %d' % num_labels


if num_labels <= 1:
    Image.open(img_path).show()
plt.show()

