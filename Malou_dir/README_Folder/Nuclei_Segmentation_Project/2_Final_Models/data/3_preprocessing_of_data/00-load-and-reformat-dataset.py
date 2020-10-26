import glob
import os
import shutil
import zipfile
import requests
from config import config_vars
import random
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm.notebook import tqdm
import skimage.io
import skimage.segmentation
import utils.dirtools
import utils.augmentation
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab

# Create output directories for transformed data

os.makedirs(config_vars["normalized_images_dir"], exist_ok=True)
os.makedirs(config_vars["boundary_labels_dir"], exist_ok=True)

config_vars["raw_images_dir"]=config_vars["home_folder"] + '/2_Final_Models/data/2_raw_images/'
config_vars["raw_annotations_dir"]=config_vars["home_folder"] + '/2_Final_Models/data/1_raw_annotations/'

# ## Normalize images

if config_vars["transform_images_to_PNG"]:

    filelist = sorted(os.listdir(config_vars["raw_images_dir"]))

    # run over all raw images
    for filename in tqdm(filelist):

        # load image and its annotation
        orig_img = skimage.io.imread(config_vars["raw_images_dir"] + filename)       

        # IMAGE

        # normalize to [0,1]
        percentile = 99.9
        high = np.percentile(orig_img, percentile)
        low = np.percentile(orig_img, 100-percentile)

        img = np.minimum(high, orig_img)
        img = np.maximum(low, img)

        img = (img - low) / (high - low) # gives float64, thus cast to 8 bit later
        img = skimage.img_as_ubyte(img) 

        skimage.io.imsave(config_vars["normalized_images_dir"] + filename[:-3] + 'png', img)    
else:
    config_vars["normalized_images_dir"] = config_vars["raw_images_dir"]

# ## Create boundary labels

filelist = sorted(os.listdir(config_vars["raw_annotations_dir"]))
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab
total_objects = 0

# run over all raw images
for filename in tqdm(filelist):

    # GET ANNOTATION
    annot = skimage.io.imread(config_vars["raw_annotations_dir"] + filename)

    # strip the first channel
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
    skimage.io.imsave(config_vars["boundary_labels_dir"] + filename, label_binary)

print("Total objects: ",total_objects)
