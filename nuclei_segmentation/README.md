# This is the directory for Filip and Malous nuclei segmentation project # 

> The bbbc folder contains reproduced code from the BBBC unet-repository.
> The inital goal has been to train a model on the BBBC images to see how it performs on aitslab images.

> The aitslab folder contains a script for converting C01 to tiff, along with a notebook that further 
> preprocesses the images and then lets the model trained on the BBBC-images make predictions on a sample of the aitslab images. 

> None of the notebooks are runnable without additional local scripts and files. See https://github.com/carpenterlab/unet4nuclei/tree/master/unet4nuclei for more details. 

Additional nuclear segmentation ground truth datasets that can be used:

https://data.broadinstitute.org/bbbc/image_sets.html

BBBC006 (both in and out of focus)

BBBC007

BBBC018

BBBC020

http://murphylab.web.cmu.edu/data/2009_ISBI_Nuclei.html
