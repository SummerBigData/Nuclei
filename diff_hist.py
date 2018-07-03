from util import *
import numpy as np
import matplotlib.pyplot as plt

img, mask = get_img(ret_mask=True)

def find_diff(img):
    max_diff = 0
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            shape = img[i-1:i+2, j-1:j+2].shape
            val = np.ones(shape) * img[i, j]
            diffs = val - img[i-1:i+2, j-1:j+2]
            max_diff = max(max_diff, np.max(diffs))
    return max_diff

for _ in range(5):
    _, ax = plt.subplots(1, 3)
    gray_imshow(ax[0], img, title='Input')
    gray_imshow(ax[2], mask, title='Target')

    max_diff = find_diff(img)
    print max_diff
    res = threshold(img, max_diff)
    gray_imshow(ax[1], res, title='thresholded')
    plt.show()

    for pt in np.argwhere(res>0):
        img[pt[0], pt[1]] = np.min(img) 
