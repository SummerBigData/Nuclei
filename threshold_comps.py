from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

import os
from os.path import join
from util import *

img_arr, total_mask = get_img(ret_mask=True)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2)

""" Get the connected components in the given image """
""" Also return the number of unique components """
def conn_comp(img):
    output = cv.connectedComponentsWithStats(img, cv.CV_32S)
    return output[0]-1, output[1]

num_labels, labels = conn_comp(total_mask)

axs[0,0].imshow(total_mask, cmap='gray')
axs[0,1].imshow(labels, cmap='jet')

print 'Number of masks: %d' % len(masks)
print 'Number of components in original: %d' % num_labels
num_labels_true = num_labels

_, img_thresh = cv.threshold(img_arr, 25, 255, cv.THRESH_BINARY)
num_labels, thresh_labels = conn_comp(img_thresh)

thresh_disp = axs[1,0].imshow(img_thresh, cmap='gray')
thresh_comp_disp = axs[1,1].imshow(thresh_labels, cmap='jet')

print 'Number of components in thresholded: %d' % num_labels


""" Change the threshold value for the axes displayed in matplotlib """
def change_thresh(thresh_val):
    _, img_thresh = cv.threshold(img_arr, thresh_val, 255, cv.THRESH_BINARY)
    num_labels, thresh_labels = conn_comp(img_thresh)
    thresh_disp.set_data(img_thresh)
    thresh_comp_disp.set_data(thresh_labels)
    print 'Threshold: %d; %d/%d' % (thresh_val, num_labels, num_labels_true)


""" Create a slider to change the threshold value interactively in mpl """
from matplotlib.widgets import Slider
slider_ax = plt.axes([0.2, 0.025, 0.65, 0.03], facecolor='lightgoldenrodyellow')
thresh_slider = Slider(slider_ax, 'Threshold', 0, 50, valinit=25)
thresh_slider.on_changed(change_thresh)

plt.show()
