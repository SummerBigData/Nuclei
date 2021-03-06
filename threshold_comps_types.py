from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

import os
from os.path import join
from util import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
fig, axs = plt.subplots(2, 2)


""" Load a random image and return it converted to black and white and its masks """
def get_img_and_mask():
    img_arr, total_mask, img_id = get_img(ret_mask=True, ret_id=True)
    img_path = join('data', img_id, 'images', img_id+'.png')
    num_mask = len(get_mask_names(img_id))
    return img_id, img_path, img, img_arr, num_mask, total_mask 


""" Return the number of connected components and the label mask for an image """
def conn_comp(img):
    output = cv.connectedComponentsWithStats(img, cv.CV_32S)
    return output[0]-1, output[1]


img_arr, img_thresh = None, None
num_labels, thresh_labels = None, None
thresh_disp, thresh_comp_disp = None, None
num_labels_true, labels_true = None, None

""" Initial loading and plotting """
def load_img():
    global img_arr, img_thresh, num_labels, thresh_labels, thresh_disp
    global thresh_comp_disp, num_labels_true, labels_true

    img_id, img_path, img, img_arr, num_mask, total_mask = get_img_and_mask()
    num_labels_true, labels_true = conn_comp(total_mask)

    """ Plot the ground truth total mask and its connected components """
    axs[0,0].imshow(total_mask, cmap='gray')
    axs[0,1].imshow(labels_true, cmap='jet')

    print 'Number of masks: %d' % num_mask
    print 'Number of components in original: %d' % num_labels_true

    """ Initially threshold the input image and find its connected components """
    _, img_thresh = cv.threshold(img_arr, 25, 255, cv.THRESH_BINARY)
    num_labels, thresh_labels = conn_comp(img_thresh)

    """ Plot the thresholded mask and its connected components """
    global thresh_disp, thresh_comp_disp
    thresh_disp = axs[1,0].imshow(img_thresh, cmap='gray')
    thresh_comp_disp = axs[1,1].imshow(thresh_labels, cmap='jet')

    print 'Number of components in thresholded: %d' % num_labels

    """ If no connect components were found in the thresholded image,
    display the original image to see what might've caused this """
    if num_labels <= 1:
        Image.open(img_path).show()



""" Change the thresold value and possible threshold type for the input image """
ttype=cv.THRESH_BINARY
def change_thresh(thresh_val):
    global img_arr, img_thresh, num_labels, thresh_labels, ttype
    _, img_thresh = cv.threshold(img_arr, thresh_val, 255, ttype)
    num_labels, thresh_labels = conn_comp(img_thresh)

    global thresh_disp, thresh_comp_disp
    thresh_disp.set_data(img_thresh)
    thresh_comp_disp.set_data(thresh_labels)
    print 'Threshold: %d; %d/%d' % (thresh_val, num_labels, num_labels_true)


""" Create a slider for the threshold value and attach change_thresh to it """
slider_ax = plt.axes([0.2, 0.025, 0.65, 0.03], facecolor='lightgoldenrodyellow')
thresh_slider = Slider(slider_ax, 'Threshold', 0, 50, valinit=25)
thresh_slider.on_changed(change_thresh)


""" Change the threshold type to BINARY """
def thresh_bin(event):
    global ttype
    ttype = cv.THRESH_BINARY
    change_thresh(thresh_slider.val)

""" Change the threshold type to TOZERO """
def thresh_tz(event):
    global ttype
    ttype = cv.THRESH_TOZERO
    change_thresh(thresh_slider.val)

""" Get a new image and repeat the display process """
def new_img(event):
    load_img() 


""" Create two buttons to change the threshold type """
ax_bin = plt.axes([0.15, 0.925, 0.15, 0.03])
bbin = Button(ax_bin, 'Binary')
bbin.on_clicked(thresh_bin)

ax_tz = plt.axes([0.35, 0.925, 0.15, 0.03])
btz = Button(ax_tz, 'To Zero')
btz.on_clicked(thresh_tz)

ax_new = plt.axes([0.55, 0.925, 0.15, 0.03])
bnew = Button(ax_new, 'New Image')
bnew.on_clicked(new_img)


load_img()
plt.show()

