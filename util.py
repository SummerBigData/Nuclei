import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from os.path import join, isfile

import numpy as np
from scipy.misc import imread, imsave
import cv2 as cv
from cv2 import fastNlMeansDenoising as mean_denoise

def arr(obj):
    return np.array(obj)

def threshold(img, tval, ttype=None):
    if ttype is None:
        ttype = cv.THRESH_BINARY
        if np.mean(img) >= 127.5:
            ttype = cv.THRESH_BINARY_INV
    return cv.threshold(img, tval, 255, ttype)[1]

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
def all_imgs(ids=None, gray=True, denoise=False, ret_ids=False, white=None):
    if ids is None:
        if white is None:
            ids = all_ids()
        else:
            white_ids = open('white_img_ids.txt').readline().split(',')
            if white:
                ids = white_ids
            else:
                ids = [id for id in all_ids() if not id in white_ids]

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
def get_img(id=None, gray=True, denoise=False, ret_mask=False, erode=False, ret_id=False, white=None):
    if id is None:
        id = get_random_id()
        if not white is None:
            white_ids = open('white_img_ids.txt').readline().split(',')
            if white:
                id = np.random.choice(white_ids)
            else:
                id = np.random.choice(arr([id for id in all_ids() if not id in white_ids]))

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

def get_test_data(just_X=False, ret_ids=False):
    from boundary_fcn import cvt_to_gray
    if just_X:
        from os import listdir
        ids = listdir('test_data1')
        X = [imread(join('test_data1', f, 'images', f+'.png')) for f in ids]
        X = cvt_to_gray(X)
        if ret_ids:
            return X, ids
        return X

    ids, y = decode_solution_file('data/stage1_solution.csv')
    X = [imread(join('test_data1', f, 'images', f+'.png')) for f in ids]
    X = cvt_to_gray(X)
    if ret_ids:
        return X, y, ids
    return X, y

from numpy.random import uniform as rand
from numpy.random import choice
from keras.preprocessing.image import ImageDataGenerator as IDG

def random_transform():
    return {
        'theta': rand(-15, 15), 'tx': rand(-25, 25), 'ty': rand(-25, 25),
        'shear': rand(-15, 15), 'zx': rand(0.65, 1.35), 'zy': rand(0.65, 1.35),
        'flip_horizontal': choice([True, False]), 'flip_vertical': choice([True, False]),
        'channel_shift_intensity': 0, 'brightness': None}

def apply_transform(img, mask, plt=None):
    dg = IDG(fill_mode='constant', cval=0)
    tr = random_transform()

    img = np.expand_dims(img/255., axis=2)
    mask = np.expand_dims(mask/255., axis=2)
     
    img = dg.apply_transform(img, tr)[:,:,0]
    mask = dg.apply_transform(mask, tr)[:,:,0]
    
    if plt:
        show_img_and_mask(img, mask, plt)
    else:
        return img, mask

def show_img_and_mask(img, mask, plt):
    _, ax = plt.subplots(1, 2)
    gray_imshow(ax[0], img, title='Image')
    gray_imshow(ax[1], mask, title='Mask')
    plt.show()

def decode_rle(df):
    w, h = int(df['Width'].iloc[0]), int(df['Height'].iloc[0])
    img = np.zeros((h, w))
    
    for _, row in df.iterrows():
        rle = row['EncodedPixels']
        terms = list(map(int, rle.split(' ')))

        for i in range(0, len(terms)-1, 2):
            start = terms[i]-1
            length = terms[i+1]
            col = start//h
            row = start % h

            for j in range(length):
                img[row+j, col] = 1
    return img

from pandas import read_csv
def decode_solution_file(fname):
    df = read_csv(fname)
    ids = set(df['ImageId'])
    imgs = []

    for id in ids:
        img_df = df[df.ImageId == id]
        imgs.append(decode_rle(img_df))

    return list(ids), imgs

def get_random_id():
    return np.random.choice(all_ids())

# Get the ids for all the masks associated with a certain image
def get_mask_names(img_id):
    dirname = join('data', img_id)
    mask_dir = join(dirname, 'masks')
    return [join(mask_dir, m) for m in os.listdir(mask_dir)]

def load_all_bboxs(img_id):
    mask_names = get_mask_names(img_id)
    bboxs = []

    for mask_name in mask_names:
        mask = imread(mask_name)
        pixels = np.argwhere(mask>0)
        t, l = pixels[:,0].min(), pixels[:,1].min()
        b, r = pixels[:,0].max(), pixels[:,1].max()
        bboxs.append([t, l, b, r])

    return arr(bboxs)
        

# Return the center of mass of every mask for the given id
from scipy.ndimage.measurements import center_of_mass as com
def load_all_centroids(img_id):
    masks = get_mask_names(img_id)
    centrs = []

    for mask in masks:
        mask_img = imread(mask)
        centrs.append(com(mask_img))

    return centrs

from skimage.morphology import label
def centroids_from_img(img):
    comps = label(img>0.5)
    for i in range(1, len(np.unique(comps))):
        stamped = np.zeros_like(img)
        stamped[comps==i] = 1
        yield com(stamped) 
    

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
    ax_obj = ax.imshow(img, cmap)
    if not title is None:
        ax.set_title(title)
    return ax_obj

import iou

def test_img(img, mask):
    return iou.iou_metric(img, mask)

def test_imgs(imgs, masks):
    return np.mean([iou.iou_metric(img, mask) for (img, mask) in zip(imgs, masks)])

def test_model(model, gen, m, patches=False, ret_ious=False, model_white=None):
    ious = []
    for _ in range(m):
        X, y = next(gen)
        if patches:
            p = model.predict(X)[:,:,:,0]
            y = y[:,:,:,0]
            ious.append(test_imgs(p, y)) 
        else:
            if not model_white is None and np.mean(X[0,:,:,0]) >= X[0,:,:,0].max()/2.:
                p = model_white.predict(X)[0,:,:,0]
            else:
                p = model.predict(X)[0,:,:,0]
            y = y[0,:,:,0]
            ious.append(test_img(p, y))
   
    mean_iou = np.mean(ious)
    if ret_ious:
        return mean_iou, ious
    return mean_iou

from scipy.misc import imresize
def batchify(x, unet=False):
    assert len(x.shape) == 2

    if unet:
        print x.shape
        new_h, new_w = x.shape
        s = [1024, 512, 256, 128]
        for size in s:
            if x.shape[0] >= size:
                new_h = size
                break
        for size in s:
            if x.shape[1] >= size:
                new_w = size
                break
        x = imresize(x, (new_h, new_w))
        print x.shape

    return x.reshape(1, x.shape[0], x.shape[1], 1)

def unbatchify(x):
    assert len(x.shape) == 4
    return x[0,:,:,0]

def load_unets(name_black, name_white=None):
    from keras.models import model_from_json

    with open(join('models', name_black, 'model.json')) as f:
        json = f.read()

    model_black = model_from_json(json)
    model_black.load_weights(join('models', name_black, 'model.h5'))

    if not name_white is None:
        with open(join('models', name_white, 'model.json')) as f:
            json = f.read()

        model_white = model_from_json(json)
        model_white.load_weights(join('models', name_white, 'model.h5'))

        return model_black, model_white

    return model_black
