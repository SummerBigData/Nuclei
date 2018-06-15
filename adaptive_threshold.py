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

#total_mask = np.array(Image.open(join(dirname, 'images', 'mask.jpeg')))
total_mask = np.zeros(Image.open(img_path).size)
for mask in os.listdir(join(dirname, 'masks')):
    total_mask += np.array(Image.open(join(dirname, 'masks', mask)))

num_mask = len(os.listdir(join(dirname, 'masks')))

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2)

def conn_comp(img):
    output = cv.connectedComponentsWithStats(img, cv.CV_32S)
    return output[0]-1, output[1]

num_labels, labels = conn_comp(total_mask)

axs[0,0].imshow(total_mask, cmap='gray')
axs[0,1].imshow(labels, cmap='jet')

print 'Number of masks: %d' % num_mask
print 'Number of components in original: %d' % num_labels
num_labels_true = num_labels

img = Image.open(img_path)
img_arr = np.array(img)
img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
img_thresh = cv.adaptiveThreshold(img_arr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
num_labels, thresh_labels = conn_comp(img_thresh)

for i in range(1, num_labels+1):
    if np.sum(thresh_labels == i) <= 5:
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

