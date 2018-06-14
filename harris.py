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

print img_id
img_arr = np.array(Image.open(img_path))
Image.open(img_path).show()

import matplotlib.pyplot as plt
f = plt.figure()

masks = os.listdir(mask_dir)
total_mask = np.zeros_like(img_arr[:,:,0])
for mask in masks:
    mask_img = Image.open(join(mask_dir, mask))
    total_mask += np.array(mask_img)

gray_mask = total_mask.copy()
total_mask = cv.cvtColor(total_mask, cv.COLOR_GRAY2BGRA)
f.add_subplot(1, 2, 1)
plt.imshow(total_mask)

corners = total_mask.copy()
gray_mask = np.float32(gray_mask)
harris = cv.cornerHarris(gray_mask, 4, 3, 0.04)
corners[harris > 0.05*harris.max()] = [0, 0, 255, 255]

f.add_subplot(1, 2, 2)
plt.imshow(corners)
plt.show(block=True)
