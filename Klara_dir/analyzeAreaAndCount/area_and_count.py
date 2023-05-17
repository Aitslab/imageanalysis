import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

import skimage.io
import skimage.morphology
import skimage.segmentation

from config import config_vars
import utils

import  utils.metrics

#define function
def get_predicted_objects(prediction, results,image_name):


    pred_objects = len(np.unique(prediction))
    if pred_objects <= 1:
        return results

    area_pred = np.histogram(prediction, bins=pred_objects)[0][1:]

    pred_objects -= 1

    print([image_name]*pred_objects)
    data = np.asarray([
        np.arange(pred_objects),[image_name]*pred_objects,
        area_pred.copy()])

    results = pd.concat([results, pd.DataFrame(data=data.T, columns=["Pred_Object","Image_name", "Area"])])

    return results

#define all images
with open('/proj/berzelius-2021-21/users/klara/Segmentation/data/4_filelists/MFGTMPcx7_170720100001_names.txt') as image_list:
    all_images = [f.strip() for f in image_list]
all_images

# Determine area and number of nuclei
config_vars["object_dilation"] =3

from skimage.color import rgb2gray,rgb2lab

print("################################################################")
print("Model number")

pred_obj = pd.DataFrame(columns=["Pred_Object","Image_name" ,"Area"])

for image_name in all_images:
    # Load predictions
    pred_filename = os.path.join("/proj/berzelius-2021-21/users/klara/Segmentation/plate_script/MFGTMPcx7_170720100001/out/segm/", image_name)
    prediction = skimage.io.imread(pred_filename)

    if len(prediction.shape) == 3:
        prediction = rgb2lab(prediction)
        prediction = prediction[:,:,0]


      # Apply object dilation
    if config_vars["object_dilation"] > 0:
        struct = skimage.morphology.square(config_vars["object_dilation"])
        prediction = skimage.morphology.dilation(prediction, struct)
    elif config_vars["object_dilation"] < 0:
        struct = skimage.morphology.square(-config_vars["object_dilation"])
        prediction = skimage.morphology.erosion(prediction, struct)

    ####################################################################################    
    #### Testing prediction with no small objects on annot and prediction #####
    #ground_truth = skimage.morphology.remove_small_objects(ground_truth, min_size=100) 
    prediction = skimage.morphology.remove_small_objects(prediction, min_size=100)
    
    pred_obj = get_predicted_objects( 
          prediction, 
          pred_obj,image_name
      )
    #####################################################################################
pred_obj.to_csv('MFGTMPcx7_170720100001.csv',index=False)
