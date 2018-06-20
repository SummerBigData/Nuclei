import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries

from util import *

imgs, ids = all_imgs(ret_ids=True, denoise=True)

for i in range(5):
    _, axs = plt.subplots(1, 2)
    
    img = imgs[np.random.choice(range(len(ids)))]
    fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    #quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

    gray_imshow(axs[0], img)
    gray_imshow(axs[1], fz)
    #gray_imshow(axs[2], quick)
    plt.show()
