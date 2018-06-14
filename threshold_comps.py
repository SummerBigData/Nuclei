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
fig, axs = plt.subplots(2, 2)

masks = os.listdir(mask_dir)
total_mask = None
for mask in masks:
    mask_img = Image.open(join(mask_dir, mask))
    if total_mask is None:
        total_mask = np.zeros_like(np.array(mask_img))
    total_mask += np.array(mask_img)

def conn_comp(img):
    output = cv.connectedComponentsWithStats(img, cv.CV_32S)
    return output[0]-1, output[1]

num_labels, labels = conn_comp(total_mask)

axs[0,0].imshow(total_mask, cmap='gray')
axs[0,1].imshow(labels, cmap='jet')

print 'Number of masks: %d' % len(masks)
print 'Number of components in original: %d' % num_labels
num_labels_true = num_labels

img = Image.open(img_path)
img_arr = np.array(img)
img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
_, img_thresh = cv.threshold(img_arr, 25, 255, cv.THRESH_BINARY)
num_labels, thresh_labels = conn_comp(img_thresh)

thresh_disp = axs[1,0].imshow(img_thresh, cmap='gray')
thresh_comp_disp = axs[1,1].imshow(thresh_labels, cmap='jet')

print 'Number of components in thresholded: %d' % num_labels

def change_thresh(thresh_val):
    _, img_thresh = cv.threshold(img_arr, thresh_val, 255, cv.THRESH_BINARY)
    num_labels, thresh_labels = conn_comp(img_thresh)
    thresh_disp.set_data(img_thresh)
    thresh_comp_disp.set_data(thresh_labels)
    print 'Threshold: %d; %d/%d' % (thresh_val, num_labels, num_labels_true)


from matplotlib.widgets import Slider
slider_ax = plt.axes([0.2, 0.025, 0.65, 0.03], facecolor='lightgoldenrodyellow')
thresh_slider = Slider(slider_ax, 'Threshold', 0, 50, valinit=25)
thresh_slider.on_changed(change_thresh)

if num_labels <= 1:
    Image.open(img_path).show()

plt.show()

