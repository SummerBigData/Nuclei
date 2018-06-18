import os
from os.path import join

import numpy as np

def arr(obj):
    return np.array(obj)

def all_ids():
    img_ids = os.listdir('data')
    img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
    return list(filter(
        lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

def get_random_id():
    return np.random.choice(all_ids)

def get_mask_names(img_id):
    dirname = join('data', img_id)
    mask_dir = join(dirname, 'masks')
    return [join(mask_dir, m) for m in os.listdir(mask_dir)]

def load_total_mask(img_id):
    dirname = join('data', img_id)
    img_path = join(dirname, 'images', img_id+'.png')

    masks = get_mask_names(img_id)
    total_mask = None
    for mask in masks:
        mask_img = Image.open(mask)
        if total_mask is None:
            total_mask = np.zeros_like(np.array(mask_img))
        total_mask += np.array(mask_img)
    return total_mask
