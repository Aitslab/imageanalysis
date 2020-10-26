import numpy as np
import pandas as pd
import skimage
from skimage import io
import skimage.segmentation
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage import img_as_ubyte
import numpy as np
from scipy.ndimage import distance_transform_edt

import sys
np.set_printoptions(threshold=sys.maxsize)

# GET ANNOTATION
with open(config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/filelist_all_aits_annotated_images.txt','r') as O1:

    for image in O1:
        image = image.rstrip()
        annot = skimage.io.imread(config_vars["home_folder"] + '/2_Final_Models/data/1_raw_annotations/' + image)

        # strip the first channel
        if annot.shape[2]!=3:
            annot = annot[:,:,0]
        else:
            annot = rgb2lab(annot)
            annot = annot[:,:,0]
        # label the annotations nicely to prepare for future filtering operation

        annot = skimage.morphology.label(annot)
        annot = annot.astype(np.uint8)
        # find boundaries
        boundaries = skimage.segmentation.find_boundaries(annot, background = 0).astype(np.uint8)

        # BINARY LABEL
        # prepare buffer for binary label
        label_binary = np.zeros((annot.shape + (3,)))

        # write binary label
        label_binary[(annot == 0) & (boundaries == 0), 0] = 1
        label_binary[(annot != 0) & (boundaries == 0), 1] = 1
        label_binary[boundaries == 1, 2] = 1

        label_binary = img_as_ubyte(label_binary)

        # save it - converts image to range from 0 to 255
        skimage.io.imsave(config_vars["home_folder"] + '/2_Final_Models/data/boundary_labels/'+ 'relabel2_' + image, label_binary)
