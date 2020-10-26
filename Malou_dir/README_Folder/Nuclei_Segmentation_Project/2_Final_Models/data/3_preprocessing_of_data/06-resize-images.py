import skimage
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize
import os
from skimage import img_as_ubyte
import skimage.segmentation
import numpy as np

def create_boundary_label(im):

    # strip the first channel
    #print(len(im.shape))
    if len(im.shape)>2:
        if im.shape[2]!=3:
            annot = im[:,:,0]
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

imagefile = open(config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/11_training.txt')
filelist = []
for line in imagefile:
    line = line.rstrip()
    filelist.append(line)
imagefile.close()
imagepath = config_vars["home_folder"] + '/2_Final_Models/data/norm_images/'
labelpath = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_raw_annotations/'
boundarypath = config_vars["home_folder"] + '/2_Final_Models/data/boundary_labels/'
for image in filelist:
    im = skimage.io.imread(imagepath + image)
    labelim = skimage.io.imread(labelpath + image)
    im = resize(im, (1104,1104))
    im = img_as_ubyte(im)
    labelim = resize(labelim, (1104,1104))
    labelim = img_as_ubyte(labelim)
    boundaryim = (create_boundary_label(labelim))
    plt.imshow(im)
    plt.show()
    plt.imshow(boundaryim)
    plt.show()
    skimage.io.imsave(imagepath + 'resized_' + image , im)  
    skimage.io.imsave(boundarypath + 'resized_'+ image, boundaryim)
