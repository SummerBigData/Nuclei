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
    masks = get_mask_names(img_id)
    boundary_img = None
    total_mask = None
    shd_save = True

    # For each mask for the current image, first erode it individually,
    # then find it border.
    #
    # Overlay this border onto the total boundaries image
    for mask in masks:
        mask_arr = imread(mask)
        
        if total_mask is None or boundary_img is None:
            boundary_img = np.zeros(mask_arr.shape)
            total_mask = np.zeros(mask_arr.shape)

        ''' Don't erode
        # Perform the erosion and find the border of the eroded mask.
        kernel = np.ones((2, 2))
        mask_ero = cv.erode(mask_arr, kernel, iterations=1)
        contours = find_contours(mask_ero, 0.0)
        '''

        contours = find_contours(mask_arr, 0.0)

        # Sometimes after erosion, the mask will be gone.
        # For now we just ignore that mask and don't save the image.
        if len(contours) == 0:
            print 'not saving'
            shd_save = False
            break

        boundary = contours[0].astype(int)
        img = np.zeros_like(boundary_img)

        # Set each boundary point to 255 in a new image to overlay
        # onto the total boundary image
        #
        # I know this is kind of ugly but it's the only way I could get it
        # to work.
        for pt in boundary:
            if pt[0] > 0:
                img[pt[0]-1, pt[1]] = 255.

            img[pt[0], pt[1]] = 255.

            if pt[0] < img.shape[0]-1:
                img[pt[0]+1, pt[1]] = 255.
        
        total_mask += mask_arr
        boundary_img += img

    if shd_save:
        imsave(join('data', img_id, 'images', 'bounds_dilated.png'), boundary_img)
