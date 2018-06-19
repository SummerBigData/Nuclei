from PIL import Image
import numpy as np
from skimage.measure import find_contours
from scipy.misc import imsave

import os
from os.path import join
from util import *

import matplotlib.pyplot as plt
import cv2 as cv

for i, img_id in enumerate(all_ids()):
    masks = get_mask_names(img_id)
    boundary_img = None
    total_mask = None

    for mask in masks:
        mask_arr = arr(Image.open(mask))  
        
        if total_mask is None or boundary_img is None:
            boundary_img = np.zeros(mask_arr.shape)
            total_mask = np.zeros(mask_arr.shape)

        boundary = find_contours(mask_arr, 0.0)[0].astype(int)

        img = np.zeros_like(boundary_img)
        for pt in boundary:
            img[pt[0], pt[1]] = 255.0
        
        kernel = np.ones((2, 2))
        img = cv.dilate(img, kernel, iterations=1)

        total_mask += mask_arr
        boundary_img += img/255.0

    imsave(join('data', img_id, 'images', 'boundaries_dilated.png'), boundary_img)
