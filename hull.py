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


""" Pick a random image and get its total mask """
img_ids = os.listdir('data')
img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
img_ids = list(filter(
    lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

img_id = np.random.choice(img_ids)
print img_id

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
            #"""
            shd_app = True
            close_pts = []

            # for every existing point off of the convex hull,
            # if it is close enough to the current point,
            # remove that point if it is closer to the hull,
            # otherwise, don't append the current point
            for i, diff in enumerate(diffs):
                if dist(diff, mv) < np.sqrt(2.):
                    if dist(diff, hv) < dist(mv, hv):
                        close_pts.append(i)
                    else:
                        shd_app = False
                        break

            if shd_app:
                diffs.append([mv[0], mv[1]])
                dists.append(dist(mv, hv))
            #"""
            #diffs.append((mv[0], mv[1]))
            #dists.append(np.sqrt((mv[0]-hv[0])**2 + (mv[1]-hv[1])**2))

    # Take the points on the mask contour that are farthest from the
    # hull contour and remove difference points that are close to them
    # on the mask contour.
    # Repeat this so that the difference points are sparse
    #idxs = np.argsort(np.array(dists))
    #for i in idxs:

    axs[0].scatter([d[1] for d in diffs], [d[0] for d in diffs], s=3.5, c='r')

    if len(diffs) > 1:
        from closestpair import closestpair
        closest = closestpair(diffs)
        axs[0].scatter([closest[0][1], closest[1][1]], [closest[0][0], closest[1][0]], s=5.0, c='g')

    plt.show()
