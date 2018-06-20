from cv2 import fastNlMeansDenoising as denoise

import numpy as np
import matplotlib.pyplot as plt

from util import *

imgs, ids = all_imgs(ret_ids=True)

for i in range(5):
    idx = np.random.choice(range(len(imgs)))
    img = imgs[idx]
    #denoised = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15, multichannel=False)
    denoised = denoise(img, 5, 7, 21)

    _, axs = plt.subplots(1, 3)
    gray_imshow(axs[0], img)
    gray_imshow(axs[1], denoised)
    gray_imshow(axs[2], load_total_mask(ids[idx]))
    plt.show()
