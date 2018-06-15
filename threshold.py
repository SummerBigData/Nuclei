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

img = Image.open(img_path)

img_arr = np.array(img)
_, img_thresh = cv.threshold(img_arr, 25, 255, cv.THRESH_BINARY)
img_thresh = cv.cvtColor(img_thresh, cv.COLOR_BGR2GRAY)

import matplotlib.pyplot as plt
f = plt.figure()

f.add_subplot(1, 2, 1)
#plt.imshow(img_thresh, cmap='gray')
plt.imshow(img_arr)

masks = os.listdir(mask_dir)
total_mask = np.zeros_like(img_arr[:,:,0])
for mask in masks:
    mask_img = Image.open(join(mask_dir, mask))
    total_mask += np.array(mask_img)

#print np.linalg.norm(img_thresh-total_mask)/np.linalg.norm(img_thresh+total_mask)
#print np.sum(img_thresh == 0)
#print np.sum(total_mask == 0)

f.add_subplot(1, 2, 2)
plt.imshow(total_mask, cmap='gray')
plt.show(block=True)
