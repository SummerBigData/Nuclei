from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

import os
from os.path import join
from util import *

img_arr, total_mask = get_img(ret_mask=True)

kernel = np.ones((3, 3))
img_arr = cv.erode(img_arr, kernel, iterations=1)
print np.mean(np.where(img_arr>0))

#_, img_thresh = cv.threshold(img_arr, 25, 255, cv.THRESH_BINARY)
img_thresh = np.zeros_like(img_arr)

w, h = 5, 5
while not img_arr.shape[0] % w == 0:
    w += 1
while not img_arr.shape[1] % h == 0:
    h += 1

for i in range(0, img_arr.shape[0]//w):
    for j in range(0, img_arr.shape[1]//h):
        patch = img_arr[i*w:(i+1)*w, j*h:(j+1)*h]
        #_, patch = cv.threshold(patch, int(np.mean(patch)), 255, cv.THRESH_BINARY)
        patch = cv.adaptiveThreshold(patch, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        img_thresh[i*w:(i+1)*w, j*h:(j+1)*h] = patch

import matplotlib.pyplot as plt
_, axs = plt.subplots(2, 2)

axs[0,0].imshow(imread(img_path))
axs[0,1].imshow(img_arr, 'gray')
axs[1,0].imshow(img_thresh, 'gray')
axs[1,1].imshow(total_mask, 'gray')
plt.show()
