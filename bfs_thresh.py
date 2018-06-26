import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from util import *

img, mask = get_img(ret_mask=True, denoise=True)

def find_maxima(x, k=3):
    res = np.zeros_like(x)
    for i in range(x.shape[0]-k+1):
        for j in range(x.shape[1]-k+1):
            patch = x[i:i+3, j:j+3]
            if np.count_nonzero(patch) == 0:
                continue
            max_pt = np.argwhere(patch == patch.max())[0]
            res[i+max_pt[0], j+max_pt[1]] = patch.max()
    return res

_, axs = plt.subplots(1, 3)
gray_imshow(axs[0], img)

res = find_maxima(img)
res = find_maxima(res)
res = find_maxima(res)
gray_imshow(axs[1], res)

#from skimage.measure import find_contours
#contours = find_contours(res, 15)
#for c in contours[:10]:
#    axs[1].plot(c[:,1], c[:,0], linewidth=2)

gray_imshow(axs[2], mask)

#pts = np.argwhere(pts>0)
#axs[1].scatter(pts[:,1], pts[:,0], s=1.5, c='r')

plt.show()
