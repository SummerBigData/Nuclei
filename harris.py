from PIL import Image
import PIL.ImageShow
PIL.ImageShow._viewers = [PIL.ImageShow._viewers[0]]

import numpy as np
import cv2 as cv

import os
from os.path import join
from util import *

img, mask, img_id = get_img(gray=True, ret_mask=True, ret_id=True)
print img_id
Image.open(img_path).show()

import matplotlib.pyplot as plt
f = plt.figure()

f.add_subplot(1, 2, 1)
plt.imshow(mask)

corners = mask.copy()
harris = cv.cornerHarris(mask, 4, 3, 0.04)
corners[harris > 0.05*harris.max()] = [255, 0, 0, 255]

f.add_subplot(1, 2, 2)
plt.imshow(corners)
plt.show(block=True)
