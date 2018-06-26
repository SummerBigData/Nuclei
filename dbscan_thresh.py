""" WARNING: Will maybe freeze the computer """
import numpy as np
import cv2 as cv

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from util import *

img, mask = get_img(ret_mask=True)

pts = np.argwhere(img>0)
X = StandardScaler().fit_transform(pts.astype(np.float64))
db = DBSCAN(eps=0.75, min_samples=3).fit(X)
labels = db.labels_
num_cluster = len(set(labels)) - (1 if -1 in labels else 0)

_, axs = plt.subplots(1, 2)
gray_imshow(axs[0], img)
gray_imshow(axs[1], mask)

colors = ['b', 'g', 'c', 'm', 'y']
for label_num, c in zip(range(1, num_cluster), colors):
    ps = pts[np.argwhere(labels == label_num).flatten()]
    axs[0].scatter(ps[:,0], ps[:,1], s=10, c=c)

plt.show()
