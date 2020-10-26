#!/usr/bin/env python
import glob
import os
import shutil
import zipfile
import requests
import random
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm.notebook import tqdm
import skimage.io
import skimage.segmentation
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab

pathways_raw = ['Fluo-N2DL-HeLa/01/','Fluo-N2DL-HeLa/02/','Fluo-N2DH-SIM+/01/','Fluo-N2DH-SIM+/02/','Fluo-N2DH-GOWT1/01/','Fluo-N2DH-GOWT1/02/']
pathways_annot =['Fluo-N2DL-HeLa/01_ST/SEG/','Fluo-N2DL-HeLa/02_ST/SEG/','Fluo-N2DH-SIM+/01_GT/SEG/','Fluo-N2DH-SIM+/02_GT/SEG/','Fluo-N2DH-GOWT1/01_ST/SEG/','Fluo-N2DH-GOWT1/02_ST/SEG/']

norm_images = config_vars["home_folder"] + '/3_data/2_additional_datasets/1_celltracking_challenge_data/normalized_images/'
boundary_labels = config_vars["home_folder"] + '/3_data/2_additional_datasets/1_celltracking_challenge_data/boundary_labels/'

os.makedirs(norm_images, exist_ok = True)
os.makedirs(boundary_labels, exist_ok = True)

total_objects = 0
index = 0
for i in range(len(pathways_raw)):
    raw_dir = config_vars["home_folder"] + '/3_data/2_additional_datasets/1_celltracking_challenge_data/' + pathways_raw[i]
    annot_dir = config_vars["home_folder"] + '/3_data/2_additional_datasets/1_celltracking_challenge_data/' + pathways_annot[i]
    filelist_raw = sorted(os.listdir(raw_dir))
    filelist_annot = sorted(os.listdir(annot_dir))
    # run over all raw images
    for filename in tqdm(filelist_raw):
        # load image and its annotation
        orig_img = skimage.io.imread(raw_dir + filename)       

        # IMAGE

        # normalize to [0,1]
        percentile = 99.9
        high = np.percentile(orig_img, percentile)
        low = np.percentile(orig_img, 100-percentile)

        img = np.minimum(high, orig_img)
        img = np.maximum(low, img)

        img = (img - low) / (high - low) # gives float64, thus cast to 8 bit later
        img = skimage.img_as_ubyte(img) 

        skimage.io.imsave(norm_images + '{}_'.format(index) + filename[:-3]  + 'png', img)  

    for filename in tqdm(filelist_annot):
        # GET ANNOTATION
        annot = skimage.io.imread(annot_dir + filename)

        # strip the first channel
        if len(annot.shape)>2:
            if annot.shape[2]!=3:
                annot = annot[:,:,0]
            else:
                annot = rgb2lab(annot)
                annot = annot[:,:,0]
        # label the annotations nicely to prepare for future filtering operation

        annot = skimage.morphology.label(annot)
        total_objects += len(np.unique(annot)) - 1

        # find boundaries
        boundaries = skimage.segmentation.find_boundaries(annot, mode = 'outer')

        # BINARY LABEL

        # prepare buffer for binary label
        label_binary = np.zeros((annot.shape + (3,)))

        # write binary label
        label_binary[(annot == 0) & (boundaries == 0), 0] = 1
        label_binary[(annot != 0) & (boundaries == 0), 1] = 1
        label_binary[boundaries == 1, 2] = 1

        label_binary = img_as_ubyte(label_binary)
        # save it - converts image to range from 0 to 255
        skimage.io.imsave(boundary_labels + '{}_'.format(index)+ 't' + filename[-7:-3] + 'png' , label_binary)
    index += 1
