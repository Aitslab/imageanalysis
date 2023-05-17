#!/usr/bin/env python
# coding: utf-8

# # Step 03
# # Predict segmentations

import os
import os.path
import skimage.io
import skimage.morphology
import tensorflow as tf
import utils.dirtools
import utils.metrics
import utils.model_builder
from config import config_vars
from skimage import img_as_ubyte
import numpy as np
import glob
#import cv2

# # Configuration

#%%

# Partition of the data to make predictions (test or validation)
partition = "test"

experiment_name = 'MFGTMPcx7_170521130001'

config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)

data_partitions = utils.dirtools.read_data_partitions(config_vars)

print(data_partitions[partition])


# In[4]:


# Device configuration

# Configuration to run on GPU
#import timeit

#start = timeit.default_timer()


#       inter_op_parallelism_threads=1)

# Configuration to run on GPU

#configuration = tf.compat.v1.ConfigProto()
#configuration.gpu_options.allow_growth = True
#configuration.gpu_options.visible_device_list = "0,1"
#session = tf.compat.v1.Session(config = configuration)
#tf.compat.v1.keras.backend.set_session(session)

#stop = timeit.default_timer()
#print('GPU set up: ', stop - start)

#with tf.compat.v1.Session() as sess:
#    devices = sess.list_devices()
#    print('devicessssssssssssssssssssssssssssssss',devices)
#global graph
#graph = tf.get_default_graph()

#from keras import backend as K
from keras.backend import clear_session



#####################################################################
def prediction_images(image_names):
    
    imagebuffer = skimage.io.imread_collection(image_names)

    images = imagebuffer.concatenate()

    

    dim1 = images.shape[1]
    dim2 = images.shape[2]

    images = images.reshape((-1, dim1, dim2,1))# config_vars["input_dimensions"]))

    # preprocess (assuming images are encoded as 8-bits in the preprocessing step)
    images = images /100#255
    
    with graph.as_default():
         predictions = model.predict(images, batch_size=1)
    
    print('after predicctionnnnnnnnnnnnnnnnnnnnnn')
    for i in range(len(imagebuffer.files)):
        filename_path = imagebuffer.files[i]
        filename = os.path.basename(filename_path)
        print(filename)
    
        probmap = predictions[i].squeeze()
        
    
    
        skimage.io.imsave(config_vars["probmap_out_dir"] + filename, (probmap).astype(float))
    
        pred = utils.metrics.probmap_to_pred(probmap, 
                                         config_vars["boundary_boost_factor"])


    
        label = utils.metrics.pred_to_label(pred,config_vars['cell_min_size'] )
    

    
        save_flag = skimage.io.imsave(config_vars["labels_out_dir"] + filename, (label).astype(int))

    

    return save_flag









####################################################################
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count



##################################################################


with open(config_vars["path_files_MFGTMPcx7_170521130001"]) as image_list:
    image_names_all = [os.path.join(config_vars["MFGTMPcx7_170521130001_dir"], f.strip()) for f in image_list]

print('imaaaaaaaaaaaaaaaaageeeeeeeeeeeeeeeeeeeeeeeeees done')
clear_session()
global model
global graph

model = utils.model_builder.get_model_3_class(1104, 1104,1)
                                              #config_vars["input_dimensions"])
model.load_weights('/proj/berzelius-2021-21/users/klara/Segmentation/plate_script/MFGTMPcx7_170521130001/model_14.hdf5')#config_vars["model_file"]) 

model._make_predict_function()
tf.Graph()
graph = tf.get_default_graph()


print(model.summary())
CPU_LIMIT=128
with ThreadPoolExecutor(min(CPU_LIMIT,cpu_count())) as executor:
    print('befoooooooooooooooooore submit function')
    futures=[executor.submit(prediction_images ,image_names_all[i_batch*16:(i_batch+1)*16]) for i_batch in range(384)]#range(len(image_names_all))] 

    for future in as_completed(futures):
        res=future.result()
        print('resssssssssssssuuuuuuuuuuuuuult: ',res)
#test = [prediction_images(image_names_all[i_batch*2:(i_batch+1)*2]) for i_batch in range(2)]  ## use this if you do not want multi-threading





