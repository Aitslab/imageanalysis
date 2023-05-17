#!/usr/bin/python

import os

config_vars = {}

config_vars["home_folder"] = '/proj/berzelius-2021-21/users/klara/Segmentation/data/'

config_vars["model_file"]='/proj/berzelius-2021-21/users/klara/data/experiments/Model_3_Malou/'

config_vars["input_dimensions"] = 1

config_vars["root_directory"] = '/proj/berzelius-2021-21/users/klara/Segmentation/plate_script/'

config_vars["learning_rate"] = 1e-4

config_vars["epochs"] = 15

config_vars['cell_min_size'] = 100

config_vars["max_training_images"] = 130

config_vars["steps_per_epoch"] = 500

config_vars["pixel_depth"] = 8

config_vars["batch_size"] = 1

config_vars["val_batch_size"] = 1

config_vars["rescale_labels"] = True

config_vars["crop_size"] = 256

config_vars["boundary_boost_factor"] = 1

config_vars["object_dilation"] = 3

config_vars["raw_annotations_dir"] = config_vars["home_folder"] + "1_raw_annotations/"
config_vars["normalized_images_dir"] = config_vars["home_folder"] + "norm_images/"
config_vars["boundary_labels_dir"] = config_vars["home_folder"] + "boundary_labels/"
config_vars["small_dir_dir"] = config_vars["home_folder"] + "small_dir/"

config_vars["MFGTMPcx7_170525010001_dir"] = config_vars["home_folder"] + "MFGTMPcx7_170525010001/"
config_vars["MFGTMPcx7_170524170002_dir"] = config_vars["home_folder"] + "MFGTMPcx7_170524170002/"
config_vars["MFGTMPcx7_170524210001_dir"] = config_vars["home_folder"] + "MFGTMPcx7_170524210001/"
config_vars["MFGTMPcx7_170525050001_dir"] = config_vars["home_folder"] + "MFGTMPcx7_170525050001/"

config_vars["MFGTMPcx7_170720000001_dir"] = config_vars["home_folder"] + "MFGTMPcx7_170720000001/"
config_vars["MFGTMPcx7_170522030001_dir"] = config_vars["home_folder"] + "MFGTMPcx7_170522030001/"
config_vars["MFGTMPcx7_170521130001_dir"] = config_vars["home_folder"] + "MFGTMPcx7_170521130001/"


config_vars["path_files_training"] = os.path.join(config_vars["home_folder"], '4_filelists/training.txt')
config_vars["path_files_validation"] = os.path.join(config_vars["home_folder"], '4_filelists/VALIDATION.txt')
config_vars["path_files_test"] = os.path.join(config_vars["home_folder"], '4_filelists/TEST.txt')
config_vars["path_files_small_dir"] = os.path.join(config_vars["home_folder"], '4_filelists/small_dir_names.txt')


config_vars["path_files_MFGTMPcx7_170525010001"] = os.path.join(config_vars["home_folder"], '4_filelists/MFGTMPcx7_170525010001_names.txt')
config_vars["path_files_MFGTMPcx7_170524170002"] = os.path.join(config_vars["home_folder"], '4_filelists/MFGTMPcx7_170524170002_names.txt')
config_vars["path_files_MFGTMPcx7_170524210001"] = os.path.join(config_vars["home_folder"], '4_filelists/MFGTMPcx7_170524210001_names.txt')
config_vars["path_files_MFGTMPcx7_170525050001"] = os.path.join(config_vars["home_folder"], '4_filelists/MFGTMPcx7_170525050001_names.txt')

config_vars["path_files_MFGTMPcx7_170720000001"] = os.path.join(config_vars["home_folder"], '4_filelists/MFGTMPcx7_170720000001_names.txt')
config_vars["path_files_MFGTMPcx7_170522030001"] = os.path.join(config_vars["home_folder"], '4_filelists/MFGTMPcx7_170522030001_names.txt')
config_vars["path_files_MFGTMPcx7_170521130001"] = os.path.join(config_vars["home_folder"], '4_filelists/MFGTMPcx7_170521130001_names.txt')

