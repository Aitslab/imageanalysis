#!/usr/bin/env python
# coding: utf-8

# # Step 03
# # Predict segmentations

import os
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import skimage.io
import skimage.morphology

import tensorflow as tf
import keras

import utils.metrics
import utils.model_builder
print(skimage.__version__)


# # Configuration

from config import config_vars

# Partition of the data to make predictions (test or validation)
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
config_vars['path_files_validation'] =config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/VALIDATION.txt'
config_vars['path_files_test'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/TEST.txt'

partition = "validation"

experiment_name = 'Model_1'

config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)

data_partitions = utils.dirtools.read_data_partitions(config_vars)
print(data_partitions[partition])


# Configuration to run on GPU
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "0"

session = tf.compat.v1.Session(config = configuration)

# apply session
tf.compat.v1.keras.backend.set_session(session)


# # Load images and run predictions


image_names = [os.path.join(config_vars["normalized_images_dir"], f) for f in data_partitions[partition]]

imagebuffer = skimage.io.imread_collection(image_names)

images = imagebuffer.concatenate()

dim1 = images.shape[1]
dim2 = images.shape[2]

images = images.reshape((-1, dim1, dim2, 1))

# preprocess (assuming images are encoded as 8-bits in the preprocessing step)
images = images / 255

# build model and load weights
model = utils.model_builder.get_model_3_class(dim1, dim2)
model.load_weights(config_vars["model_file"])

# Normal prediction time
predictions = model.predict(images, batch_size=1)


for i in range(len(images)):

    filename = imagebuffer.files[i]
    filename = os.path.basename(filename)

    probmap = predictions[i].squeeze()
    
    skimage.io.imsave(config_vars["probmap_out_dir"] + filename, probmap)
    
    pred = utils.metrics.probmap_to_pred(probmap, config_vars["boundary_boost_factor"])
    
    label = utils.metrics.pred_to_label(pred,config_vars['cell_min_size'] )

    skimage.io.imsave(config_vars["labels_out_dir"] + filename, label)




