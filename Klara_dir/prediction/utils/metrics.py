import numpy as np
import skimage.segmentation
import skimage.io
#import keras.backend as K
import tensorflow as tf
import keras as K


debug = False

def channel_precision(channel, name):
    def precision_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal( K.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_tmp, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
    
        return precision
    precision_func.__name__ = name
    return precision_func


def channel_recall(channel, name):
    def recall_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal( K.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:,:,:,channel], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
    
        return recall
    recall_func.__name__ = name
    return recall_func


## PROBMAP TO CONTOURS TO LABEL

def probmap_to_contour(probmap, threshold = 0.5):
    # assume 2D input
    outline = probmap >= threshold
    
    return outline

def contour_to_label(outline, image):
    # see notebook contours_to_labels for why we do what we do here
    
    # get connected components
    labels = skimage.morphology.label(outline, background=1)
    skimage.morphology.remove_small_objects(labels, min_size = 100, in_place = True)
    
    n_ccs = np.max(labels)

    # buffer label image
    filtered_labels = np.zeros_like(labels, dtype=np.uint16)

    # relabel as we don't know what connected component the background has been given before
    label_index = 1
    
    # start at 1 (0 is contours), end at number of connected components
    for i in range(1, n_ccs + 1):

        # get mask of connected compoenents
        mask = labels == i

        # get mean
        mean = np.mean(np.take(image.flatten(),np.nonzero(mask.flatten())))

        if(mean > 50/255):
            filtered_labels[mask] = label_index
            label_index = label_index + 1
            
    return filtered_labels


## PROBMAP TO PRED TO LABEL

def probmap_to_pred(probmap, boundary_boost_factor):
    # we need to boost the boundary class to make it more visible
    # this shrinks the cells a little bit but avoids undersegmentation
    pred = np.argmax(probmap * [1, 1, boundary_boost_factor], -1)
    
    return pred


def pred_to_label(pred, cell_min_size, cell_label=1):
    # Only marks interior of cells (cell_label = 1 is interior, cell_label = 2 is boundary)
    cell=(pred == cell_label)
    # fix cells
    cell = skimage.morphology.remove_small_holes(cell, area_threshold=cell_min_size)
    cell = skimage.morphology.remove_small_objects(cell, min_size=cell_min_size)
    
    # label cells only
    [label, num] = skimage.morphology.label(cell, return_num=True)
    return label

