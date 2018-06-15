from PIL import Image
import numpy as np

import os
from os.path import join

import matplotlib.pyplot as plt

img_ids = os.listdir('data')
img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
img_ids = list(filter(
    lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

for i in range(5):
    img_id = np.random.choice(img_ids)
    dirname = join('data', img_id)
    img_path = join(dirname, 'images', img_id+'.png')
    mask_path = join(dirname, 'images', 'mask.jpeg')

    img_arr = np.array(Image.open(img_path))
    mask_arr = np.array(Image.open(mask_path))

    fig = plt.figure()
    
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_arr)

    fig.add_subplot(1, 2, 2)
    plt.imshow(mask_arr, cmap='gray')

    plt.show() 
