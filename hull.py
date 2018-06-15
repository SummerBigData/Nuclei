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

import matplotlib.pyplot as plt

masks = os.listdir(mask_dir)
#idx = np.random.randint(0, len(masks))
#total_mask = np.array(Image.open(join(mask_dir, masks[idx])))
total_mask = None
for mask in masks:
    mask_img = Image.open(join(mask_dir, mask))
    if total_mask is None:
        total_mask = np.zeros_like(np.array(mask_img))
    total_mask += np.array(mask_img)

output = cv.connectedComponentsWithStats(total_mask, cv.CV_32S)
num_label, labels = output[0], output[1]

label_num = np.random.randint(1, num_label)

for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if labels[i,j] == label_num:
            labels[i,j]= 1.0
        else:
            labels[i,j] = 0.0

total_mask = labels*255

from skimage.morphology import convex_hull_image
from skimage.measure import find_contours 

hull = convex_hull_image(total_mask)
contour = find_contours(hull, 0.0)[0]
mask_contour = find_contours(total_mask, 0.0)[0]

_, axs = plt.subplots(1, 2)
axs[0].imshow(total_mask, cmap='gray')
axs[1].imshow(hull, cmap='gray')
#axs[1].plot(contour[:,1], contour[:,0], lw=1.5)
axs[1].plot(mask_contour[:,1], mask_contour[:,0], lw=1.5)

hull_area = np.sum(hull == 1)
mask_area = np.sum(total_mask == 255)
print mask_area, hull_area

diffs = []
for hv, mv in zip(contour, mask_contour):
    xmatch = mv[0] <= hv[0]+1 and mv[0] >= hv[0]-1
    ymatch = mv[1] <= hv[1]+1 and mv[1] >= hv[1]-1
    if not (xmatch and ymatch):
        diffs.append(mv)
print diffs
axs[0].scatter([d[1] for d in diffs], [d[0] for d in diffs], s=2.5, c='r')

plt.show()
