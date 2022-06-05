Overview:

This is the datasets we have used for training the model
----------------------------------------------------------------------------------------------------

Dataset Description:

Each ground truth file is stored as a .mat file, with the keys:
'inst_map'
'inst_centroid'
 
'inst_map' is a 1000x1000 array containing a unique integer for each individual nucleus. i.e the map ranges from 0 to N, where 0 is the background and N is the number of nuclei.

'inst_centroid' is a Nx2 array, giving the x and y coordinates of the centroids of each instance (in order of inst map ID).

To extract the patches needed when training the model, run the extract_patches.py file with its paths modified.

The folder structure in the Test and Train folders are correct for the model to run
------------------------------------------------------------------------------------------
