from PIL import Image
import numpy as np

import os
from os.path import join

img_ids = os.listdir('data')
img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
img_ids = list(filter(
    lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

for img_id in img_ids:
    dirname = join('data', img_id)
    img_path = join(dirname, 'images', img_id+'.png')
    mask_dir = join(dirname, 'masks')

    masks = os.listdir(mask_dir)

    img_arr = np.array(Image.open(img_path))
    print img_arr.shape

    total_mask = np.zeros_like(img_arr[:,:,0])
    for mask in masks:
        mask_img = Image.open(join(mask_dir, mask))
        total_mask += np.array(mask_img)

    img = Image.fromarray(total_mask, 'L')
    img.save(join(dirname, 'images', 'mask.jpg'))
