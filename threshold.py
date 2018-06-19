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
_, axs = plt.subplots(1, 3)

masks = os.listdir(mask_dir)
total_mask = np.zeros_like(img_arr[:,:,0])
for mask in masks:
    mask_img = Image.open(join(mask_dir, mask))
    total_mask += np.array(mask_img)

#print np.linalg.norm(img_thresh-total_mask)/np.linalg.norm(img_thresh+total_mask)
#print np.sum(img_thresh == 0)
#print np.sum(total_mask == 0)

#plt.imshow(img_thresh, cmap='gray')
img_arr = cv.cvtColor(img_arr, cv.COLOR_BGRA2GRAY)
axs[0].imshow(img_arr, 'gray')

axs[1].imshow(img_thresh, 'gray')
axs[2].imshow(total_mask, 'gray')
#axs[2].imshow(np.array(Image.open(join(dirname, 'images', 'boundaries.png'))), 'gray')
plt.show()
