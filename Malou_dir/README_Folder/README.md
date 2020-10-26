# README 






1. Nuclei Segmentation
    
    - 1.1 Setup
        
        - Folder setup
        - Environment setup
    
    - 1.2 Load images

    - 1.3 Annotate images using cvat

    - 1.4 Preprocessing of images

         - Convert images to png
         - Create training/validation/test fraction text files
         - Normalization
         - Creating border labels
         - Augmentation (affine transformation)
         - Preprocessing of image-sets for Model 5

    - 1.5 Training

    - 1.6 Prediction

    - 1.7 Evaluation
    
    - 1.8 Models

# 1 Nuclei Segmentation

I have with help from the script by Broad Bioimage Benchmarc Collection (BBBC), found at https://github.com/carpenterlab/unet4nuclei, recreated their experiment with the use of images from Aits lab, and additional images from other open source datasets.

In the BBBC repository, the full script can be found for their experiments, but for our purpose the script has been modified. 

Some modifications had to be done due to outdated versions of python packages, and some modifications because of different image-formats.

This README will give instructions of how to recreate the experiments I've made.

## 1.1 Setup

### Folder setup

To be able to run the scripts without doing major changes in the script, the folder structure should look the same as in this project.

In the script **config.py**, the variable 
```python
config_vars["home_folder"] = '/home/maloua/Malou_Master/5_Models'
```
must be changed to your absolute pathway to where you will store this project. The subfolders should have the following structure:

Folder structure:

- **2_Final_Models**
    - **1_Model1**
        - 02-training.py
        - 03-prediction.py
        - 04-evaluation.ipynb
        - config.py
        - **utils**
            - augmentation.py
            - data_provider.py
            - dirtools.py
            - evaluation.py
            - experiment.py
            - metrics.py
            - model_builder.py
            - objectives.py

    - **data**
        - **1_raw_annotations**
        - **2_raw_images**
        - **3_preprocessing_of_data**
            - 00-load-and-reformat-dataset.py
            - 01-Augmentation.py
            - 05-Augment-validation-set.py
            - 06-resize-images.py
            - Relabeling_1.py
            - Relabeling_2.py
            - config.py
        - **4_filelists**
            - 1-2_training.txt
            - 3_training.txt
            - 4_training.txt
            - 5_training.txt
            - 5_training_500.txt
            - 6_training.txt
            - 7_training.txt
            - 8_training.txt
            - 10_training.txt
            - 11_training.txt
            - 12_training.txt
            - 13_training.txt
            - TEST.txt
            - VALIDATION.txt
        - **utils**
            - augmentation.py
            - data_provider.py
            - dirtools.py
            - evaluation.py
            - experiment.py
            - metrics.py
            - model_builder.py
            - objectives.py
- **3_data**
    - **2_additional_datasets**
        - **1_celltracking_challenge_data**
        - **2_BBBC_image_sets**


In the experiment several models are created. The folder structure for each model is exactly the same as **1_Model1**, and for each model added, an identical folder is added named **#_Model#** where "#" is replaced with the model number. Each model has some modifications done in the scripts which is specified in section **1.7 - Models**

### Environment setup

## 1.2 Load images

 
### Download BBBC images

The images from the bbbc experiment can be found here:

https://data.broadinstitute.org/bbbc/BBBC039/images.zip

The images are extracted and put in the folder **2_Final_Models/data/2_raw_images**

And the annotations for the images:

https://data.broadinstitute.org/bbbc/BBBC039/masks.zip

The mask images are extracted and put in the folder **2_Final_Models/data/1_raw_annotations**
Downloading image sets for Model 5

### Additional datasets



**Celltracking challenge data:**

<u> GFP-GOWT1 mouse stem cells:</u>

http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

<u> HeLa cells stably expressing H2b-GFP: </u>

http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip

<u> Simulated nuclei of HL60 cells stained with Hoescht:</u>

http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip

**BBBC image sets:**

<u> kaggle-dsbowl-2018-dataset:</u>

https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes/archive/master.zip

<u> BBBC Murine bone-marrow derived macrophages: </u>

https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip

https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_nuclei.zip

The zipfiles should be extracted to the following folderstructure:

- **3_data**
    - **2_additional_datasets**
        - **1_celltracking_challenge_data**
            - Fluo-N2DL-HeLa
            - Fluo-N2DH-SIM+
            - Fluo-N2DH-GOWT1
        - **2_BBBC_image_sets**
            - kaggle-dsbowl-2018-dataset-fixes
            - BBBC020_v1_images
            - BBBC020_v1_outlines_nuclei

    


**Aits lab images**

We have worked with 100 randomly selected images from the full data set consisting of approx. 6 million images, these images are of the format .C01 which is a microscopy format. The input images in the BBBC scripts are expected to be .tiff or .png, so all 100 images are first converted to png.

For this I have created a script that will take a directory as input, and output the converted images to a new selected directory.

The script is done with argparse, and can be used just by downloading the script and following these steps:


**Downloading and installing bftools:**
```bash
cd ~/bin
wget http://downloads.openmicroscopy.org/latest/bio-formats/artifacts/bftools.zip
unzip bftools.zip
rm bftools.zip
export PATH=$PATH:~/bin/bftools
```
**Installing required python packages:**

```bash
pip install argparse
pip install os
pip install subprocess
pip install tqdm
pip install pathlib
```


**The program is run like:**

```bash
python3 format_convertion.py -i INDIR -o OUTDIR -ift IN_FILETYPE -oft OUT_FILETYPE
```




## 1.3 Annotate images using cvat

We have annotated 50 images to use in the experiment, we have used the annotation program cvat. Information about the program and installation instructions are found on their github page, https://github.com/opencv/cvat.

Only one label is used for annotation, nucleus, and each nucleus is drawn with the polygon tool.

The work is saved using the option “DUMP ANNOTATION” -> “MASK ZIP 1.1”

That will create and download a zip file with one folder of images only with the class (nucleus), showing all nucleus in the same color, and one folder with annotations of the objects, each image will be an RGB image, with all objects being different colors to distinguish between them.

In the creation of our labels, the object images was used. The images should be extracted to the folder These images are extracted to the folder **2_Final_Models/data/1_raw_annotations**

These images are not of the same format as the bbbc institute’s, so the script had to be modified to fit these images.

The images we have annotated are the ones with filenames found in **1-2_training.txt**, **VALIDATION.txt** and **TEST.txt**.

## 1.4 Preprocessing of images

### Normalization and creation of boundary labels

First the images are normalized to have pixel values between 0-1 instead of 0-255, and converted to png if that is not already done.

Then boundary labels are created. Objects are found in the annotation image using the skimage module, both for finding distinguished objects, and for finding boundaries of the objects. 

These steps are done in the script **00-load-and-reformat-dataset.py**

The annotations are expected to be one image with all annotations, where each object is of different pixelvalue. for the images downloaded for Model 5, some additional preprocessing was needed.

### Preprocessing of the images for Model 5

#### kaggle-dsbowl-2018-dataset:

This datasets consists of different images, many images that is not similar to our dataset, and not wanted in our model. It is no specific structure of the directories and where to find the similar images, but the images similar to ours are all grey scale, and can be distinguished in that way.

Another difference with this dataset is that the annotations are separated per object, so that it exists one image per object instead of one image with all objects.

To extract the images a script was created, it goes through the directories, it is one image per directory. The image is checked if it is grey scale or not. The image type is RGB-D even if it is grey scale, so to control if it is grey scale the pixel values are compared. If the image is grey scale, the first 3 values of each pixel are expected to be the same (the fourth value is the value of the image transparancy).

If the image is gray scale then the masks are extracted and combined to form one image. Each mask image is black and white, the object is white. The mask images are combined, with each mask given a different pixel value.

In the same script the normalization and boundarylabeling are done in the same way as the previous datasets.

The script is named **Preprocessing_and_merging_annotations_BBBC038.py**.

After running the script and looking through the images, some images were not suited for our model, so they were removed. It was done by creating a list with all the images to be removed and the following bash command:

Create a doc folder:
```bash
mkdir 3_data/2_additional_datasets/2_BBBC_image_sets/doc
cd 3_data/2_additional_datasets/2_BBBC_image_sets/doc
```

Put the textfile in the created doc folder, and execute the following command:
```bash
cat filelist_wrong_images.txt | while read line; do rm ../BBBC038_*/$line; done
```


#### BBBC Murine bone-marrow derived macrophages:

Each folder in **3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_v1_images** consists of 3 subfolders with images, one with the cells, one with the nucleus, and one with the combined images. We are only interested in the images of the nucleus, and need to extract those. These images are the ones with the ending _c5.TIF in their names.

The images needs to be converted to grey images, and the mask images needs to be combined as in the BBBC038 dataset.

This is all done with the script **Preprocessing_BBBC020.py**

#### Celltracking challenge images (GFP-GOWT1 mouse stem cells, HeLa cells stably expressing H2b-GFP and Simulated nuclei of HL60 cells stained with Hoescht)

These images need no specific preprocessing. A script was created to extract, normalize and create boundary labels for all the images in one go.

The script is named **Preprocessing_celltracking_images.py**

Creating 5_training.txt:

For Model 5, all these images should be used in the training set, together with the images used in Model 4. The below bash commands will create the file 5_training.txt with all the images, and place it in the folder **2_Final_Models/data/4_filelists**.
```bash
ls 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_normalized_images/ > 2_Final_Models/data/4_filelists/5_training.txt && 
ls 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_normalized_images/ >> 2_Final_Models/data/4_filelists/5_training.txt && 
ls 3_data/2_additional_datasets/1_celltracking_challenge_data/normalized_images/ >> 2_Final_Models/data/4_filelists/5_training.txt &&
```
For Model 5 we want 500 additional images, but we have 1368 images. To use only 500 images the file 5_training.txt is randomly sorted and the 500 first lines are used and put in a new textfile using bashcommand:

```bash
sort -R 2_Final_Models/data/4_filelists/5_training.txt | head -n 500 > 2_Final_Models/data/4_filelists/5_training_500.txt
```
The list we got and used for this can be found under **1.10 - Docs**

Model 5 should also include the images used in Model 4, so the lines from **4_training.txt** should be added to **5_training_500.txt**. It is done with the bash command:

```bash
cat 2_Final_Models/data/4_filelists/4_training.txt >> 2_Final_Models/data/4_filelists/5_training_500.txt
```

#### Moving all data to the same folder:

After doing the above steps with the preprocessing, all images are ready to use in the model for training. The images should be moved to **2_Final_Models/data/boundary_labels** and **2_Final_Models/data/norm_images**. These folders were created during the preprocessing step when running the script **00-load-and-reformat-dataset.py**

Moving the files is done using bash command:
```bash
mv 3_data/2_additional_datasets/1_celltracking_challenge_data/boundary_labels/* 2_Final_Models/data/boundary_labels/

mv 3_data/2_additional_datasets/1_celltracking_challenge_data/normalized_images/* 2_Final_Models/data/norm_images/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_boundary_labels/* 2_Final_Models/data/boundary_labels/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_normalized_images/* 2_Final_Models/data/norm_images/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_boundary_labels/* 2_Final_Models/data/boundary_labels/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_normalized_images/* 2_Final_Models/data/norm_images/
```

### Augmentation

To increase the size of the dataset, we have done augmentations on the images. Some augmentations are implemented in the training step. In some models we have used affine transformation on the images, which is not done in the training step, which creates an image that is slightly transformed but is interpreted as a different image to the neural network. The original script was provided from the BBBC group, but needed some adjustments.

For Model 1, the 30 images will be augmented with affine transformation, which is done using script **01-Augmentation.py**.

For Model 2.2 augmentation was also done for the images in the validation set, which was not done in **01-Augmentation.py**, the script for the validation set is the same with some small changes, the full script is **05-Augment-validation-set.py**.


## 1.5 Training



## 1.6 Prediction

In the prediction step, the model is loaded and predicts the images in the validation set. The images generated from the prediction are red, green and blue images, as the boundary label images. From the prediction images objects are identified by the green area. A True and False array is created with the value True for all green pixels, and False everywhere else. The skimage.morphology.label module will then create a label image from this, that will be stored and later used in the evaluation step.

This is done using script **03-prediction.py**.


## 1.7 Evaluation

## 1.8 Models

* **Model 1** - Using 40 images from Aits lab. 30 images for training, and 10 for validation.
* **Model 2** - Using 40 images from Aits lab, with elastic transformation done on the training images so a total of 300 images is used as the training set.
* **Model 3** - The same images as Model 2, with 100 additional images from the BBBC039 image set.
* **Model 4** - The same images as Model 2, with 200 additional images from the BBBC039 image set.
* **Model 5** - The same as Model 4, with additional 500 randomly selected images from the image collections BBBC038, BBBC020, "GFP-GOWT1 mouse stem cells", "HeLa cells stably expressing H2b-GFP" and "Simulated nuclei of HL60 cells stained with Hoescht"
* **Model 6** - Using 2 images from Aits lab
* **Model 7** - Using 5 images from Aits lab
* **Model 8** - Using 15 images from Aits lab
* **Model 9** - Model 1 trained for additional 15 epochs
* **Model 10** - Same images as Model 1 and 29 additional images from the BBBC038 image set, handpicked based on similarity to aits images.
* **Model 11** - Using only the 29 handpicked images used in Model 10.
* **Model 12** - Same as Model 10, with the 29 additional images resized to the same size as the images from Aits lab (1104x1104).
* **Model 13** - 40 Aits lab images with affine transformation + 333 images from BBBC038
* **Model 14** - Model 3 trained for 15 additional epochs
* **Model 15** - Model 5 trained for 15 additional epochs
* **Model 16** - Model 10 trained for 15 additional epochs
* **Model 17** - Model 1, with the weight parameter of the boundary label changed from 10 to 5. No dilation in the evaluation step
* **Model 18** - Model 1, with modification to the weight parameter of the boundary label. backround, interior and boundary now has the same value instead of 1/1/10. No dilation done in the evaluation step now.
* **Model 19** - Model 13 trained for 15 additional epochs
* **Model 20** - Same images as Model 1, but boundary labels created differently, objects are increased in size and then applied 2 pixel boundaries, one inside of object, and one outside.
* **Model 21** - Same images as Model 1, but 2 pixel boundaries.
* **Model 22** - Model 20 trained for 5 additional epochs
* **Model 24** - Model created with same images as Model 1, but with boundaries created as the original BBBC model (2 pixels boundary). Objects less than 40 pixels removed before training.
* **Model 25** - Model created with image set 2, and objects less than 25 pixels removed before training.
* **Model 26** - Model created with image set 2, and objects less than 100 pixels removed before training.


For all models, the variable "experiment_name" needs to be changed to the corresponding model name. e.g. for Model 5 ```experiment_name = 'Model_5'```. The variable is found in script **02-training.py**, **03-prediction.py** and **04-evaluation.ipynb**


For model 3, 4, 5 and 13, where affine transformation is done on the images, following line in the script **02-training.py** needs to be changed:
```python
data_partitions = utils.dirtools.read_data_partitions(config_vars, load_augmented = False)
```
changes to :
```python
data_partitions = utils.dirtools.read_data_partitions(config_vars)
```

### Additional changes in corresponding model :

#### Model 3:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/3_training.txt'
```

#### Model 4:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/4_training.txt'
```

#### Model 5:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/5_training_500.txt'
```

#### Model 6:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/6_training.txt'
```

#### Model 7:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/7_training.txt'
```

#### Model 8:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/8_training.txt'
```

#### Model 9:
in script **02-training.py**
under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_1/model.hdf5')
```


#### Model 10
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/10_training.txt'
```

#### Model 11
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/5_Models/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/11_training.txt'
```
#### Model 12
For this model some images were resized, which was done using the script **06-resize-images.py**, found in section **1.9 - Scripts**.


in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/12_training.txt'
```

#### Model 13
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/13_training.txt'
```


#### Model 14:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/3_training.txt'
```
and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_3/model.hdf5')
```
#### Model 15:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + /2_Final_Models/data/4_filelists/5_training.txt'
```

and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_5/model.hdf5')
```

#### Model 16:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/10_training.txt'
```

and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_10/model.hdf5')
```

#### Model 17

In the utils script **objectives.py** the following line is changed:

```python
class_weights = tf.constant([[[[1., 1., 10.]]]])
```
to:

```python
class_weights = tf.constant([[[[1., 1., 5.]]]])
```
and in the script **04-evaluation.ipynb** the following section is removed:

```python
# Apply object dilation
if config_vars["object_dilation"] > 0:
    struct = skimage.morphology.square(config_vars["object_dilation"])
    prediction = skimage.morphology.dilation(prediction, struct)
elif config_vars["object_dilation"] < 0:
    struct = skimage.morphology.square(-config_vars["object_dilation"])
    prediction = skimage.morphology.erosion(prediction, struct)
```

#### Model 18

In the utils script **objectives.py** the following line is changed:

```python
class_weights = tf.constant([[[[1., 1., 10.]]]])
```
to:

```python
class_weights = tf.constant([[[[1., 1., 1.]]]])
```

and in the script **04-evaluation.ipynb** the following section is removed:

```python
# Apply object dilation
if config_vars["object_dilation"] > 0:
    struct = skimage.morphology.square(config_vars["object_dilation"])
    prediction = skimage.morphology.dilation(prediction, struct)
elif config_vars["object_dilation"] < 0:
    struct = skimage.morphology.square(-config_vars["object_dilation"])
    prediction = skimage.morphology.erosion(prediction, struct)
```

#### Model 19:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/13_training.txt'
```

and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_13/model.hdf5')
```

#### Model 20:
For this model new boundary labels were created using the script **Relabeling_1.py**

In script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/relabel_1-2_training.txt'
```
#### Model 21:
For this model new boundary labels were created using the script **Relabeling_2.py**

In script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/relabel2_1-2_training.txt'
```

#### Model 22:
In **config.py**

the variable 
```python
config_vars["epochs"] = 15
```
is changed to:

```python
config_vars["epochs"] = 5
```


In script **02-training.py**

Under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_20/model.hdf5')
```
