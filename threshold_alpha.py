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

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3)

mask_img = Image.open(join(dirname, 'images', 'mask.jpeg'))

img = Image.open(img_path)
img_arr = np.array(img)

img_arr_al = img_arr[:,:,-1]
img_arr_bw = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

_, img_thresh_bw = cv.threshold(img_arr_bw, 25, 255, cv.THRESH_BINARY)
_, img_thresh_al = cv.threshold(img_arr_al, 25, 255, cv.THRESH_BINARY)

axs[0,0].imshow(img_arr)
axs[0,1].imshow(img_arr_bw, cmap='gray')
axs[0,2].imshow(img_arr_al, cmap='gray')
axs[1,0].imshow(img_thresh_bw, cmap='gray')
axs[1,1].imshow(img_thresh_al, cmap='gray')
axs[1,2].imshow(np.array(mask_img), cmap='gray')

plt.show()

