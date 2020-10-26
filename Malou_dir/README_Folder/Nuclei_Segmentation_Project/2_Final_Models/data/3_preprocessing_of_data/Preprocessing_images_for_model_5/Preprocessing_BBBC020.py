#!/usr/bin/env python


import os
import skimage.io
import skimage.segmentation
import matplotlib.pyplot as plt
from PIL import Image,ImageChops
import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab
from skimage.color import rgb2grey


def create_boundary_label(im):

    # strip the first channel
    if len(im.shape)>2:
        if im.shape[2]!=3:
            annot = annot[:,:,0]
        else:
            im = rgb2lab(annot)
            im = annot[:,:,0]
    # label the annotations nicely to prepare for future filtering operation

    im = skimage.morphology.label(im)
    #print(np.unique(im))
    # find boundaries
    boundaries = skimage.segmentation.find_boundaries(im, mode = 'outer')


    label_binary = np.zeros((im.shape + (3,)))
    # write binary label
    label_binary[(im == 0) & (boundaries == 0), 0] = 1
    label_binary[(im != 0) & (boundaries == 0), 1] = 1
    label_binary[boundaries == 1, 2] = 1
    #print(np.unique(label_binary.reshape(-1, merged.shape[2]), axis=0))
    label_binary = img_as_ubyte(label_binary)
    return(label_binary)


# Normalization script
def normalize(orig_img):
        # normalize to [0,1]
        percentile = 99.9
        high = np.percentile(orig_img, percentile)
        low = np.percentile(orig_img, 100-percentile)

        img = np.minimum(high, orig_img)
        img = np.maximum(low, img)

        img = (img - low) / (high - low) # gives float64, thus cast to 8 bit later
        img = skimage.img_as_ubyte(img) 

        return(img)   


dir_path = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_v1_images/'
masks_path = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBC020_v1_outlines_nuclei/'



norm_images = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_normalized_images/'
boundary_labels = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_boundary_labels/'

os.makedirs(norm_images, exist_ok = True)
os.makedirs(boundary_labels, exist_ok = True)

image_dirs = os.listdir(dir_path)
masks_dir = os.listdir(masks_path)


for dirs in image_dirs:
    images = os.listdir(dir_path + dirs)
    for image in images:
        ending = image.split('_')[-1]
        if ending == 'c5.TIF':
            blueim = skimage.io.imread(dir_path + dirs + '/' + image)
            grayim = rgb2grey(blueim)
            imagename = image.split('.')[0]
            mask_list = []
            index = 1
            for masks in masks_dir:
                if masks.startswith(imagename):
                    mask = skimage.io.imread(masks_path + masks)
                    mask[mask==255]=index
                    mask_list.append(mask)
                    index += 1
            maskmerged = sum(mask_list)
            if isinstance(maskmerged,np.ndarray):
                grayim = normalize(grayim)
                skimage.io.imsave(norm_images + image[:-3] + 'png', grayim)
                label_mask = create_boundary_label(maskmerged)
                skimage.io.imsave(boundary_labels + image[:-3] + 'png', label_mask)
