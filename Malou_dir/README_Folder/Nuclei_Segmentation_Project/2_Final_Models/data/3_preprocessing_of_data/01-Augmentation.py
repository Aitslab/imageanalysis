import os
import pathlib
from tqdm.notebook import tqdm
import skimage.io
import skimage.segmentation
from config import config_vars
import utils.dirtools
import utils.augmentation

config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
config_vars['path_files_validation'] =config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/VALIDATION.txt'
config_vars['path_files_test'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/TEST.txt'


# ## Augment (elastic transformation)

config_vars["augment_images"] = True
def generate_augmented_examples(filelist, n_augmentations, n_points, distort, dir_boundary_labels, dir_images_normalized_8bit):

    updated_filelist = []

    # run over all raw images
    for filename in tqdm(filelist):
        print("Augmenting {}".format(filename))

        # check if boundary labels were calculated 
        my_file = pathlib.Path(dir_boundary_labels + filename)

        if my_file.is_file():

            # load image 
            x = skimage.io.imread(dir_images_normalized_8bit + filename)

            # load annotation 
            y = skimage.io.imread(dir_boundary_labels + filename)

            for n in range(1,n_augmentations):
                # augment image and annotation 
                x_augmented, y_augmented = utils.augmentation.deform(x, y, points = n_points, distort = distort)

                # filename for augmented images
                filename_augmented = os.path.splitext(filename)[0] + '_aug_{:03d}'.format(n) + os.path.splitext(filename)[1]
                skimage.io.imsave(dir_images_normalized_8bit + filename_augmented, x_augmented)
                skimage.io.imsave(dir_boundary_labels + filename_augmented, y_augmented)
                updated_filelist.append(filename_augmented)

    return filelist + updated_filelist 

if config_vars["augment_images"]:

    tmp_value = config_vars["max_training_images"]
    config_vars["max_training_images"] = 0
    tmp_partitions = utils.dirtools.read_data_partitions(config_vars, load_augmented_training = False, load_augmented_validation = False)

    training_files = generate_augmented_examples(
        tmp_partitions["training"], 
        config_vars["elastic_augmentations"], 
        config_vars["elastic_points"], 
        config_vars["elastic_distortion"], 
        config_vars["boundary_labels_dir"], 
        config_vars["normalized_images_dir"]
    )

    config_vars["max_training_images"] = tmp_value
