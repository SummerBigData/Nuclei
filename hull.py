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

""" Find the clusters of difference points on the mask boundary """
def find_clusters(diffs)
    # Normalize diffs so that they can be inputted to DBSCAN
    X = StandardScaler().fit_transform(diffs)

    # Perform DBSCAN clustering on the points off of the hull 
    db = DBSCAN(eps=0.75, min_samples=2).fit(X)
    labels = db.labels_
    num_cluster = len(set(labels)) - (1 if -1 in labels else 0)

    # If we have more than one cluster of points found, then plot the 
    # clusters and find the closest two points between any cluster 
    diffs = np.array([[d[0], d[1]] for d in diffs])
    diffs = np.fliplr(diffs)

    return num_cluster, labels, diffs

""" Find the closest pair of difference points between two clusters """
def find_closest_pair(diffs, labels, num_cluster):
    closest_pair = []
    closest_dist = np.inf

    # For each cluster, for all other clusters, find the minimum distance between
    # all pairs of points in the two clusters. Only keep the minimum of all of those
    # distances. Very brute-forced 
    for i in range(num_cluster):
        # Set of points in the first cluster
        s1 = diffs[np.argwhere(labels == i).flatten()]

        for j in range(i+1, num_cluster):
            # Set of points in the second cluster
            s2 = diffs[np.argwhere(labels == j).flatten()]

            # Distance between every pair of points in the two sets
            ds = arr([[dist(p1, p2) for p2 in s2] for p1 in s1])

            # Pair that has the smallest distance between the two sets
            idx1, idx2 = np.argmin(ds, axis=0)[0], np.argmin(ds, axis=1)[0]
            pair = (s1[idx1], s2[idx2])

            # Replace the closest_pair if the distance is smaller than closest_dist
            if dist(pair[0], pair[1]) < closest_dist:
                closest_dist = dist(pair[0], pair[1])
                closest_pair = pair

    return closest_pair


concave_masks = get_all_concave_masks(total_mask)
for mask, mask_contour, hull, hull_contour in concave_masks:
    # Make a two-panel plot:
    # On the left, plot the mask, and clusters of difference points
    # On the right, plot the mask boundary overlayed on the convex hull
    _, axs = plt.subplots(1, 2)
    gray_imshow(axs[0], mask)
    gray_imshow(axs[1], hull)
    axs[1].plot(mask_contour[:,1], mask_contour[:,0], lw=1.5)
    axs[1].plot(hull_contour[:,1], hull_contour[:,0], lw=1.5)

    # Find the points on the mask boundary that are off of
    # the boundary of the convex hull in order to cut the cluster
    diffs = find_diffs(mask_contour, hull_contour)
    axs[0].scatter([d[1] for d in diffs], [d[0] for d in diffs], s=3.5, c='r')

    # Only find the clusters if there are at least two difference points
    if len(diffs) > 1:
        num_cluster, labels, diffs = find_clusters(diffs)

        # Only plot the clusters if more than one was found
        if num_cluster > 1:
            colors = ['b', 'g', 'c', 'm', 'y']
            for label_num, c in zip(range(num_cluster), colors):
                pts = diffs[np.argwhere(labels == label_num).flatten()]
                axs[0].scatter(pts[:,0], pts[:,1], s=10, c=c)

            p1, p2 = find_closest_pair(diffs, labels, num_cluster)
            axs[0].scatter([p1[0], p2[0]], [p1[1], p2[1]], s=25, c='r')

    plt.show()
