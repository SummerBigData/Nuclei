from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

import os
from os.path import join

img_ids = os.listdir('data')
img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
img_ids = list(filter(
    lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

img_id = np.random.choice(img_ids)
dirname = join('data', img_id)
img_path = join(dirname, 'images', img_id+'.png')
mask_dir = join(dirname, 'masks')

from scipy.misc import imread
img_arr = imread(img_path)
img_arr = cv.cvtColor(img_arr, cv.COLOR_BGRA2GRAY)

kernel = np.ones((3, 3))
img_arr = cv.erode(img_arr, kernel, iterations=1)
print np.mean(np.where(img_arr>0))

#_, img_thresh = cv.threshold(img_arr, 25, 255, cv.THRESH_BINARY)
#img_thresh = cv.cvtColor(img_thresh, cv.COLOR_BGR2GRAY)
#_, img_arr = cv.threshold(img_arr, 2, 0, cv.THRESH_BINARY_INV)
img_thresh = np.zeros_like(img_arr)

w, h = 5, 5
while not img_arr.shape[0] % w == 0:
    w += 1
while not img_arr.shape[1] % h == 0:
    h += 1
print w, h
for i in range(0, img_arr.shape[0]//w):
    for j in range(0, img_arr.shape[1]//h):
        patch = img_arr[i*w:(i+1)*w, j*h:(j+1)*h]
        #print np.mean(patch)
        #_, patch = cv.threshold(patch, int(np.mean(patch)), 255, cv.THRESH_BINARY)
        patch = cv.adaptiveThreshold(patch, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        img_thresh[i*w:(i+1)*w, j*h:(j+1)*h] = patch

import matplotlib.pyplot as plt
_, axs = plt.subplots(2, 2)

masks = os.listdir(mask_dir)
#total_mask = np.zeros_like(img_arr[:,:,0])
total_mask = np.zeros_like(img_arr)
for mask in masks:
    total_mask += imread(join(mask_dir, mask))

#print np.linalg.norm(img_thresh-total_mask)/np.linalg.norm(img_thresh+total_mask)
#print np.sum(img_thresh == 0)
#print np.sum(total_mask == 0)

#plt.imshow(img_thresh, cmap='gray')
axs[0,0].imshow(imread(img_path))
axs[0,1].imshow(img_arr, 'gray')
axs[1,0].imshow(img_thresh, 'gray')
axs[1,1].imshow(total_mask, 'gray')
#axs[2].imshow(np.array(Image.open(join(dirname, 'images', 'boundaries.png'))), 'gray')
plt.show()
