from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

from skimage.morphology import convex_hull_image
from skimage.measure import find_contours 

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import os
from os.path import join
from util import *

import matplotlib.pyplot as plt

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
        hull = convex_hull_image(stamped)

        # If the area of the mask is less than 95% that of
        # the convex hull, we say that it is concave
        hull_area = np.sum(hull == 1)
        mask_area = np.sum(stamped == 255)

        if float(mask_area)/hull_area < 0.95:
            # Zoom in on the mask for better viewing
            pixels = stamped>0
            hor = np.argwhere(pixels==1)[:,1]
            ver = np.argwhere(pixels==1)[:,0]

            p = 35
            l, r = min(hor)-p, max(hor)+p
            t, b = min(ver)-p, max(ver)+p
            l, r = max(0, l), min(stamped.shape[1], r)
            t, b = max(0, t), min(stamped.shape[0], b)
            
            stamped = stamped[t:b, l:r]
            hull = hull[t:b, l:r]
            mask_contour = find_contours(stamped, 0.0)[0]
            hull_contour = find_contours(hull, 0.0)[0]

            concave_masks.append((stamped, mask_contour, hull, hull_contour))

    return concave_masks

""" Euclidean distance between two points """
def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

""" Find the points on the mask contour different from those on the hull """
def find_diffs(mask_contour, hull_contour):
    diffs = []

    # Find every point on the mask contour that is greater than or equal to
    # 1 unit away from any point on the hull contour 
    for mv in mask_contour:
        min_dist = min([dist(pt, mv) for pt in hull_contour])
        if min_dist >= np.sqrt(2.):
            diffs.append((mv[0], mv[1]))

    return diffs

""" Find the closest pair of difference points between two clusters """
def find_closest_pair(defects):
    closest_pair = []
    closest_dist = np.inf

    for i in range(len(defects)):
        for j in range(i+1, len(defects)):
            pair = (defects[i], defects[j])
            distance = dist(pair[0], pair[1])

            if distance < closest_dist:
                closest_dist = distance
                closest_pair = pair

    return closest_pair

from scipy.optimize import curve_fit
""" Find a polynomial smoothly connecting the closest points """
def find_cut(s1, s2):
    x = np.append(s1[:,0], s2[:,0])
    y = np.append(s1[:,1], s2[:,1])

    if len(x) < 3:
        def func(x, a, b):
            return np.polyval([a, b], x)
    elif len(x) == 3:
        def func(x, a, b, c):
            return np.polyval([a, b, c], x)
    elif len(x) == 4:
        def func(x, a, b, c, d):
            return np.polyval([a, b, c, d], x)
    else:
        def func(x, a, b, c, d, e):
            return np.polyval([a, b, c, d, e], x) 

    coef, _ = curve_fit(func, x, y)
    p = np.poly1d(coef)

    xp = np.linspace(min(x), max(x), 100)
    plt.plot(x, y, '.', xp, p(xp), '-')


from skimage.measure import regionprops

def split(stamped):
    cnt = cv.findContours(stamped, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[1][0]
    hull = cv.convexHull(cnt, returnPoints=False)

    defects = cv.convexityDefects(cnt, hull)
    if defects is None or len(defects) == 0:
        return None

    dists = defects[:, 0, -1]
    defects = arr([cnt[f][0] for f in defects[:, 0, 2]])
    max_dist = dists.max()

    tmp_defects = []
    for d, defect in zip(dists, defects):
        if float(d)/max_dist > 0.2:
            tmp_defects.append(defect)

    defects = arr(tmp_defects)
    if len(defects) < 2:
        return None

    pair = find_closest_pair(defects)

    split_img = stamped.copy()
    p1, p2 = pair
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    if p2[0] == p1[0]:
        m, b = 0, 1
    else:
        m = float(p2[1]-p1[1])/(p2[0]-p1[0])
        b = p2[1] - m*p2[0]

    for i in range(p2[0]-p1[0]+1):
        x = p1[0]+i
        y = int(m*x + b)
        split_img[y, x] = 0

    return split_img, defects.tolist(), pair

def is_convex(stamped):
    props = regionprops(stamped, cache=False)[0]
    return props.filled_area/float(props.convex_area) > 0.935
        
def label_and_split(tlabels, tlabel_num, bbox=True, size=None):
    stamped = np.zeros_like(total_mask)
    if not bbox and not size is None:
        stamped = np.zeros((size))
    stamped[tlabels == tlabel_num] = 1
    if is_convex(stamped):
        return

    #'''
    if bbox:
        pixels = stamped>0
        hor = np.argwhere(pixels==1)[:,1]
        ver = np.argwhere(pixels==1)[:,0]

        p = 50
        l, r = min(hor)-p, max(hor)+p
        t, b = min(ver)-p, max(ver)+p
        l, r = max(0, l), min(stamped.shape[1], r)
        t, b = max(0, t), min(stamped.shape[0], b)
        stamped = stamped[t:b, l:r]
    #'''
    
    res = split(stamped)
    if res is None:
        return
    split_img, defects, pair = res

    '''
    while True:
        print len(defects), len_def
        if len(defects) == len_def:
            break
        
        len_def = len(defects)
        labels = label(split_img)
        print '%d components' % len(np.unique(labels))

        for label_num in range(1, len(np.unique(labels))):
            split_stamped = np.zeros_like(split_img)
            split_stamped[labels == label_num] = 1
            if is_convex(split_stamped):
                continue

            res = split(split_stamped)
            if res is None:
                continue

            split_img, curr_defects, curr_pair = res
            defects.extend(curr_defects)
            pairs.append(curr_pair)
    '''

    _, ax = plt.subplots(1, 2)

    for i in range(len(defects)):
        ax[0].scatter([defects[i][0]], [defects[i][1]], c='r', s=15)

    ax[0].scatter([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], c='g', s=25)
    gray_imshow(ax[0], stamped, title='Original')
    gray_imshow(ax[1], split_img, title='Split')
    plt.show()

    sp_labels = label(split_img)
    print '%d components' % len(np.unique(sp_labels))
    for lnum in range(1, len(np.unique(sp_labels))):
        label_and_split(sp_labels, lnum, bbox=False, size=sp_labels.shape)


labels = label(total_mask)
print '%d components' % len(np.unique(labels))

for label_num in range(1, len(np.unique(labels))):
    label_and_split(labels, label_num)
