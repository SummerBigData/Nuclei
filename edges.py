from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv
from scipy.misc import imread

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
img_arr = imread(img_path)

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
_, axs = plt.subplots(1, 2)

masks = os.listdir(mask_dir)
total_mask = np.zeros_like(img_arr[:,:,0])
kernel = np.ones((3, 3))
for mask in masks:
    total_mask += cv.erode(imread(join(mask_dir, mask)), kernel, iterations=1)

img_arr = cv.cvtColor(img_arr, cv.COLOR_BGRA2GRAY)

t1, t2 = 100, 200
#edges = cv.Canny(img_arr, t1, t2)
edges = cv.Laplacian(img_arr, cv.CV_64F)

edge_ax = axs[0].imshow(edges, 'gray')
mask_ax = axs[1].imshow(total_mask, 'gray')

"""
def change_t1(t1_val):
    t1 = t1_val
    edges = cv.Canny(img_arr, t1, t2)
    edge_ax.set_data(edges)

def change_t2(t2_val):
    t2 = t2_val
    edges = cv.Canny(img_arr, t1, t2)
    edge_ax.set_data(edges)

t1_ax = plt.axes([0.115, 0.115, 0.75, 0.03])
t2_ax = plt.axes([0.115, 0.085, 0.75, 0.03])

t1_sl = Slider(t1_ax, 't1', 0, 255, valinit=100)
t2_sl = Slider(t2_ax, 't2', 0, 255, valinit=200)

t1_sl.on_changed(change_t1)
t2_sl.on_changed(change_t2)
"""

def new_img(event):
    img_id = np.random.choice(img_ids)
    img_path = join('data', img_id, 'images', img_id+'.png')
    mask_dir = join('data', img_id, 'masks')

    img_arr = imread(img_path)
    masks = os.listdir(mask_dir)

    total_mask = np.zeros_like(img_arr[:,:,0])
    kernel = np.ones((3, 3))
    for mask in masks:
        total_mask += cv.erode(imread(join(mask_dir, mask)), kernel, iterations=1)

    img_arr = cv.cvtColor(img_arr, cv.COLOR_BGRA2GRAY)
    mask_ax.set_data(img_arr)
    
    #edges = cv.Canny(img_arr, t1, t2)
    edges = cv.Laplacian(img_arr, cv.CV_64F)
    edge_ax.set_data(edges)

ax_new = plt.axes([0.75, 0.055, 0.15, 0.03])
b_new = Button(ax_new, 'New')
b_new.on_clicked(new_img)

plt.show()
