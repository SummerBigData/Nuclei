from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

from scipy.spatial import ConvexHull

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
#img_arr = np.array(Image.open(img_path))
#Image.open(img_path).show()

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3)

masks = os.listdir(mask_dir)
idx = np.random.randint(0, len(masks))
total_mask = np.array(Image.open(join(mask_dir, masks[idx])))
#total_mask = np.zeros_like(img_arr[:,:,0])
#for mask in masks:
#    mask_img = Image.open(join(mask_dir, mask))
#    total_mask += np.array(mask_img)

gray_mask = total_mask.copy()
axs[0].imshow(total_mask, cmap='gray')

corners = total_mask.copy()
corners = cv.cvtColor(corners, cv.COLOR_GRAY2BGRA)
gray_mask = np.float32(gray_mask)
harris = cv.cornerHarris(gray_mask, 4, 3, 0.04)
corners[harris > 0.05*harris.max()] = [0, 0, 255, 255]

axs[1].imshow(corners)

corner_gray = np.zeros_like(harris)
corner_gray[harris > 0.05*harris.max()] = 255

axs[2].imshow(corner_gray, cmap='gray')

plt.show()
