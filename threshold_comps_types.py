from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

import os
from os.path import join

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
fig, axs = plt.subplots(2, 2)

""" Load all of the image ids and filter those that do have masks """
img_ids = os.listdir('data')
img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
img_ids = list(filter(
    lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))


""" Load a random image and return it converted to black and white and its masks """
def get_img_and_mask():
    img_id = np.random.choice(img_ids)
    dirname = join('data', img_id)
    img_path = join(dirname, 'images', img_id+'.png')
    mask_dir = join(dirname, 'masks')
    masks = os.listdir(mask_dir)

    total_mask = None
    for mask in masks:
        mask_img = Image.open(join(mask_dir, mask))
        if total_mask is None:
            total_mask = np.zeros_like(np.array(mask_img))
        total_mask += np.array(mask_img)
    
    img = Image.open(img_path)
    img_arr = np.array(img)
    img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
    return img_id, img, img_arr, len(masks), total_mask 


""" Return the number of connected components and the label mask for an image """
def conn_comp(img):
    output = cv.connectedComponentsWithStats(img, cv.CV_32S)
    return output[0]-1, output[1]



""" Initial loading and plotting """
img_id, img, img_arr, num_mask, total_mask = get_img_and_mask()
num_labels, labels = conn_comp(total_mask)

""" Plot the ground truth total mask and its connected components """
axs[0,0].imshow(total_mask, cmap='gray')
axs[0,1].imshow(labels, cmap='jet')

print 'Number of masks: %d' % num_mask
print 'Number of components in original: %d' % num_labels
num_labels_true = num_labels

""" Initially threshold the input image and find its connected components """
_, img_thresh = cv.threshold(img_arr, 25, 255, cv.THRESH_BINARY)
num_labels, thresh_labels = conn_comp(img_thresh)

""" Plot the thresholded mask and its connected components """
thresh_disp = axs[1,0].imshow(img_thresh, cmap='gray')
thresh_comp_disp = axs[1,1].imshow(thresh_labels, cmap='jet')

print 'Number of components in thresholded: %d' % num_labels



""" Change the thresold value and possible threshold type for the input image """
def change_thresh(thresh_val, ttype=cv.THRESH_BINARY):
    _, img_thresh = cv.threshold(img_arr, thresh_val, 255, ttype)
    num_labels, thresh_labels = conn_comp(img_thresh)
    thresh_disp.set_data(img_thresh)
    thresh_comp_disp.set_data(thresh_labels)
    print 'Threshold: %d; %d/%d' % (thresh_val, num_labels, num_labels_true)


""" Create a slider for the threshold value and attach change_thresh to it """
slider_ax = plt.axes([0.2, 0.025, 0.65, 0.03], facecolor='lightgoldenrodyellow')
thresh_slider = Slider(slider_ax, 'Threshold', 0, 50, valinit=25)
thresh_slider.on_changed(change_thresh)


""" Change the threshold type to BINARY """
def thresh_bin(event):
    change_thresh(thresh_slider.val, ttype=cv.THRESH_BINARY)

""" Change the threshold type to TOZERO """
def thresh_tz(event):
    change_thresh(thresh_slider.val, ttype=cv.THRESH_TOZERO)


""" Create two buttons to change the threshold type """
ax_bin = plt.axes([0.15, 0.925, 0.15, 0.03])
bbin = Button(ax_bin, 'Binary')
bbin.on_clicked(thresh_bin)

ax_tz = plt.axes([0.35, 0.925, 0.15, 0.03])
btz = Button(ax_tz, 'To Zero')
btz.on_clicked(thresh_tz)

if num_labels <= 1:
    Image.open(img_path).show()

plt.show()

