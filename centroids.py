import numpy as np
import matplotlib.pyplot as plt
from cv2 import THRESH_TOZERO as tz

from util import *

img, id = get_img(ret_id=True, denoise=False)
img = threshold(img, int(1.2*np.mean(img)), ttype=tz)

centroids = load_all_centroids(id)
centrs = arr([[c[1], c[0]] for c in centroids])
print '%d centroids' % len(centrs)

from skimage.morphology import label
comps = label(img > int(1.0*np.mean(img)))
#comps = label(img)
print '%d components' % (len(np.unique(comps))-1)

from scipy.ndimage.measurements import center_of_mass as com
comp_centrs = arr(list(map(
    lambda c: [c[1], c[0]],
    [com(img * (comps==i)) 
        for i in range(1, len(np.unique(comps))-1)])))

_, ax = plt.subplots(1, 2)
gray_imshow(ax[0], img)
ax[0].scatter(centrs[:,0], centrs[:,1], s=10, c='r')

#gray_imshow(ax[1], comps, cmap='jet')
gray_imshow(ax[1], img)
ax[1].scatter(comp_centrs[:,0], comp_centrs[:,1], s=10, c='b')

#gray_imshow(ax[2], comps, cmap='jet')
plt.show()
