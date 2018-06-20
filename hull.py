from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

from skimage.morphology import convex_hull_image
from skimage.measure import find_contours 

import os
from os.path import join

import matplotlib.pyplot as plt

from util import *

_, total_mask, img_id = get_img(ret_mask=True, ret_id=True)
print img_id

""" Stamp just the given label number out of the components image """
def stamp(labels, label_num):
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j] == label_num:
                labels[i,j]= 1.0
            else:
                labels[i,j] = 0.0
    return labels*255


""" Return an array of all predicted concave masks out of the total image """
def get_all_concave_masks(img):
    output = cv.connectedComponentsWithStats(total_mask, cv.CV_32S)
    num_label, labels = output[0], output[1]

    concave_masks = []

    for label_num in range(1, num_label):
        # Stamp the connected component out of the total image
        stamped = stamp(labels.copy(), label_num)

        # Find the convex hull image of the cluster
        # as well as the boundaries of it and its corresponding hull
        hull = convex_hull_image(stamped)
        hull_contour = find_contours(hull, 0.0)[0]
        mask_contour = find_contours(stamped, 0.0)[0]

        # If the area of the mask is less than 95% that of
        # the convex hull, we say that it is concave
        hull_area = np.sum(hull == 1)
        mask_area = np.sum(stamped == 255)

        if float(mask_area)/hull_area < 0.95:
            concave_masks.append((stamped, mask_contour, hull, hull_contour))

    return concave_masks


concave_masks = get_all_concave_masks(total_mask)
for mask, mask_contour, hull, hull_contour in concave_masks:
    """ Make a two-panel plot:
    On the left, plot the cluster
    on the right, plot the mask boundary overlayed on the convex hull """
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(mask, cmap='gray')
    axs[1].imshow(hull, cmap='gray')
    axs[1].plot(mask_contour[:,1], mask_contour[:,0], lw=1.5)


    """ Find the points on the mask boundary that are off of
    the boundary of the convex hull in order to cut the cluster """
    diffs = []
    dists = []

    def dist(p1, p2):
        return np.sqrt(np.sum(p1-p2)**2)

    for hv, mv in zip(hull_contour, mask_contour):
        #xmatch = mv[0] <= hv[0]+1 and mv[0] >= hv[0]-1
        #ymatch = mv[1] <= hv[1]+1 and mv[1] >= hv[1]-1
        #if not (xmatch and ymatch):
        if dist(mv, hv) >= np.sqrt(2.):
            diffs.append((mv[0], mv[1]))
            dists.append(np.sqrt((mv[0]-hv[0])**2 + (mv[1]-hv[1])**2))

    axs[0].scatter([d[1] for d in diffs], [d[0] for d in diffs], s=3.5, c='r')

    if len(diffs) > 1:
        from closestpair import closestpair
        closest = closestpair(diffs)
        axs[0].scatter([closest[0][1], closest[1][1]], [closest[0][0], closest[1][0]], s=5.0, c='g')

    plt.show()
