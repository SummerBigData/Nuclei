from PIL import Image
import numpy as np
from skimage.measure import find_contours
from scipy.misc import imsave, imread

import os
from os.path import join
from util import *

#import matplotlib.pyplot as plt
import cv2 as cv

for i, img_id in enumerate(all_ids()):
    #_, axs = plt.subplots(1, 2)

    masks = get_mask_names(img_id)
    boundary_img = None
    total_mask = None
    shd_save = True

    for mask in masks:
        mask_arr = arr(Image.open(mask))  
        
        if total_mask is None or boundary_img is None:
            boundary_img = np.zeros(mask_arr.shape)
            total_mask = np.zeros(mask_arr.shape)

        kernel = np.ones((2, 2))
        mask_ero = cv.erode(mask_arr, kernel, iterations=1)
        contours = find_contours(mask_ero, 0.0)
        if len(contours) == 0:
            shd_save = False
            break
        boundary = contours[0].astype(int)

        img = np.zeros_like(boundary_img)
        for pt in boundary:
            img[pt[0], pt[1]] = 255.0
        
        #kernel = np.ones((2, 2))
        #img = cv.dilate(img, kernel, iterations=1)

        total_mask += mask_arr
        boundary_img += img

    #img = imread(join('data', img_id, 'images', img_id+'.png'))
    #img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    #axs[0].imshow(img, 'gray')
    #axs[1].imshow(boundary_img, 'gray')
    #plt.show()
    #if i == 2:
    #    break

    if shd_save:
        imsave(join('data', img_id, 'images', 'bounds_eroded.png'), boundary_img)
