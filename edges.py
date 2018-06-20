from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv
from scipy.misc import imread

import os
from os.path import join
from util import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
_, axs = plt.subplots(1, 2)


""" Initially load and display the edges and eroded mask of a random image """
img_arr, total_mask = get_img(ret_mask=True, erode=True)
edges = cv.Laplacian(img_arr, cv.CV_64F)

edge_ax = axs[0].imshow(edges, 'gray')
mask_ax = axs[1].imshow(total_mask, 'gray')


""" Load a new image and display the edges and the eroded mask """
""" in the current mpl figure """
def new_img(event):
    img_arr, total_mask = get_img(ret_mask=True, erode=True)
    edges = cv.Laplacian(img_arr, cv.CV_64F)

    edge_ax.set_data(edges)
    mask_ax.set_data(total_mask)

ax_new = plt.axes([0.75, 0.055, 0.15, 0.03])
b_new = Button(ax_new, 'New')
b_new.on_clicked(new_img)

plt.show()
