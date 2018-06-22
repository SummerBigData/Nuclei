import os
from os.path import join, isfile

import numpy as np
from scipy.misc import imread, imsave
import cv2 as cv
from cv2 import fastNlMeansDenoising as mean_denoise

def arr(obj):
    return np.array(obj)

# Return all of the image ids that have masks associated with them
def all_ids():
    img_ids = os.listdir('data')
    img_ids = list(filter(lambda f: not f.endswith('zip'), img_ids))
    return list(filter(
        lambda f: os.path.isdir(join('data', f, 'masks')), img_ids))

# Load the denoised image at the given path or, if it doesn't exit,
# denoise the original image, save the result, and return it
def load_or_denoise_and_save(path, id):
    if isfile(path):
        return imread(path)
    else:
        denoised = mean_denoise(imread(join(dir, id+'.png')), 5, 7, 21)
        imsave(path, denoised)
        return denoised

# Return numpy arrays of all images. If gray=True, then return them in grayscale
def all_imgs(ids=None, gray=True, denoise=False, ret_ids=False):
    if ids is None:
        ids = all_ids()

    # Denoising each image every time is expensive, therefore if we are denoising,
    # then hope we have each denoised image saved and try and load those.
    #
    # Otherwise, denoise and save that for next time because again,
    # screw your hard drive.
    if denoise:
        imgs = []
        for i, id in enumerate(ids):
            dir = join('data', id, 'images')
            path = join(dir, id+'_denoised.png')
            imgs.append(load_or_denoise_and_save(path, id)) 

        if gray:
            imgs = [cv.cvtColor(img, cv.COLOR_BGRA2GRAY) for img in imgs]

        if ret_ids:
            return imgs, ids
        return imgs 

    imgs = [imread(join('data', id, 'images', id+'.png')) for id in ids]

    if gray:
        imgs = [cv.cvtColor(img, cv.COLOR_BGRA2GRAY) for img in imgs]

    if denoise:
        imgs = [mean_denoise(img, 5, 7, 21) for img in imgs]

    if ret_ids:
        return imgs, ids
    return imgs

# Return an image and variuos other properties of it:
#
# id - if None, then a random image is chosen
# gray - if True, then the original image is converted to grayscale
# denoise - if True, then the original returned denoised
# ret_mask - whether or not to return the corresponding mask for the given image
# erode - whether or not to erode the mask, only matters if ret_mask=True
# ret_id - whether or not to return the corresponding id for the image chosen
#          only matters if id=None
def get_img(id=None, gray=True, denoise=False, ret_mask=False, erode=False, ret_id=False):
    if id is None:
        id = get_random_id()

    dir = join('data', id, 'images')
    if denoise:
        img = load_or_denoise_and_save(join(dir, id+'_denoised.png'), id) 
    else:
        img = imread(join(dir, id+'.png'))

    if gray:
       img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

    if not ret_mask:
        if ret_id:
            return img, id
        return img

    fname = 'mask.png'
    if erode:
        fname = 'mask_eroded.png'
    mask = load_or_create_mask(join(dir, fname), id, erode=erode)

    if ret_id:
        return img, mask, id
    return img, mask

def get_random_id():
    return np.random.choice(all_ids())

# Get the ids for all the masks associated with a certain image
def get_mask_names(img_id):
    dirname = join('data', img_id)
    mask_dir = join(dirname, 'masks')
    return [join(mask_dir, m) for m in os.listdir(mask_dir)]

# Load all the masks for a given image all overlayed on one another
#
# If erode=True, then each individual mask will be eroded before being
# overlayed so that borders between overlapping masks are defined
def load_total_mask(img_id, erode=False):
    dirname = join('data', img_id)
    img_path = join(dirname, 'images', img_id+'.png')
    size = imread(img_path).shape
    size = (size[0], size[1])

    masks = get_mask_names(img_id)
    total_mask = np.zeros(size)

    for mask in masks:
        mask_img = imread(mask)
        if erode:
            kernel = np.ones((2, 2))
            mask_img = cv.erode(mask_img, kernel, iterations=1)
        total_mask += mask_img

    return total_mask

# Either load the mask for the given id or create is and save the result
def load_or_create_mask(path, id, erode=False):
    # If the given mask already exists, just load it.
    if isfile(path):
        return imread(path)

    # Otherwise, create the mask on the fly
    # and save it for next time because screw our hard drive.
    else:
        mask = load_total_mask(id, erode=erode)
        imsave(jpath, mask)
        return mask

def all_recursive_masks(ret_ids=False, smoothed=False):
    ids = all_ids()

    fname = 'thresh_recursive.png'
    if smoothed:
        fname = 'sm_rec_mask.png'

    def path(id):
        return join('data', id, 'images', fname)

    ids = [id for id in ids if isfile(path(id))]
    masks = [imread(path(id)) for id in ids]

    if ret_ids:
        return masks, ids
    return masks

# Return all of the total masks for each of the given list of ids
# Erode the masks if erode=True.
def masks_for(ids, erode=False):
    masks = []
    fname = 'mask.png'
    if erode:
        fname = 'mask_eroded.png'

    for id in ids:
        dir = join('data', id, 'images')
        masks.append(load_or_create_mask(join(dir, fname), id, erode=erode))

    return masks

def mask_for(id, erode=False):
    fname = 'mask.png'
    if erode:
        fname = 'mask_eroded.png'

    return load_or_create_mask(join('data', id, 'images', fname), id, erode=erode)

def gray_imshow(ax, img, cmap='gray', title=None):
    ax.axis('off')
    ax.imshow(img, cmap)
    if not title is None:
        ax.set_title(title)
