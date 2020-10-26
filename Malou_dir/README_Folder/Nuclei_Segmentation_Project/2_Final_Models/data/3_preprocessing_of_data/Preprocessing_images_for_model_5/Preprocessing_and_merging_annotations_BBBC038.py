#!/usr/bin/env python
import os
import skimage.io
import skimage.segmentation
import matplotlib.pyplot as plt
from PIL import Image,ImageChops
import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab
from tqdm import tqdm


filepath = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/kaggle-dsbowl-2018-dataset-fixes-master/stage1_train/'
filelist = os.listdir(filepath)
image_storage = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_normalized_images/'
annotation_storage = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_boundary_labels/'

os.makedirs(image_storage, exist_ok= True)
os.makedirs(annotation_storage, exist_ok = True)

def create_boundary_label(im):

    if len(im.shape)>2:
        if im.shape[2]!=3:
            annot = annot[:,:,0]
        else:
            im = rgb2lab(annot)
            im = annot[:,:,0]
    # label the annotations nicely to prepare for future filtering operation

    im = skimage.morphology.label(im)

    boundaries = skimage.segmentation.find_boundaries(im, mode = 'outer')


    label_binary = np.zeros((im.shape + (3,)))
    # write binary label
    label_binary[(im == 0) & (boundaries == 0), 0] = 1
    label_binary[(im != 0) & (boundaries == 0), 1] = 1
    label_binary[boundaries == 1, 2] = 1
    label_binary = img_as_ubyte(label_binary)
    return(label_binary)

for directory in tqdm(filelist):
    imgindex = 0
    imname = os.listdir(filepath + directory + '/images')
    im = Image.open(filepath + directory + '/images/'+ imname[0])
    pix_vals = list(im.getdata())
    pix_vals = pix_vals[0:5]
    grey = True
    for pixels in pix_vals:
        if pixels[0]==pixels[1]==pixels[2]:
            grey = True
        else:
            grey = False
    if grey == True:
        im = skimage.io.imread(filepath + directory + '/images/'+ imname[0])
        if len(im.shape)>= 3:
            im = im[:,:,0]
        skimage.io.imsave(image_storage + imname[0],im)

        masks_path = filepath + directory + '/masks/'
        masks = os.listdir(masks_path)

        mask_list = []
        index = 1
        for mask in masks:
            mask_im = skimage.io.imread(masks_path + mask)
            mask_im[mask_im==255]=index
            mask_list.append(mask_im)
            index += 1
        maskmerged = sum(mask_list)
        label_mask = create_boundary_label(maskmerged)
        skimage.io.imsave(annotation_storage + imname[0], label_mask)
