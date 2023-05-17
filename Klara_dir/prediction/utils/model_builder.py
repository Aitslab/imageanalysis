#!/usr/bin/env python3   
import tensorflow as tf
import keras.layers
import keras.models

#from   keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate,Activation


CONST_DO_RATE = 0.5

option_dict_conv = {"activation": "relu", "border_mode": "same"}
option_dict_bn = {"mode": 0, "momentum" : 0.9}


# returns a core model from gray input to 64 channels of the same size
def get_core(dim1, dim2, input_dim):
    
    x = keras.layers.Input(shape=(dim1, dim2, input_dim))

    a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(x)  
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    
    y = keras.layers.MaxPooling2D()(a)

    b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    
    y = keras.layers.MaxPooling2D()(b)

    c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    
    y = keras.layers.MaxPooling2D()(c)

    d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    
    # UP

    d = keras.layers.UpSampling2D()(d)

    y = keras.layers.merge.concatenate([d, c], axis=3)

    e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.UpSampling2D()(e)

    
    y = keras.layers.merge.concatenate([e, b], axis=3)

    f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.UpSampling2D()(f)

    
    y = keras.layers.merge.concatenate([f, a], axis=3)

    y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    return [x, y]
'''
def get_new_core(dim1, dim2, input_dim):  #not used this one
    
    x = Input(shape=(dim1, dim2, input_dim))
    
    a = Conv2D(64, (3, 3), activation="relu", padding="same")(x)  
    a = BatchNormalization(momentum=0.9)(a)

    a = Conv2D(64, (3, 3), activation="relu", padding="same")(a)
    a = BatchNormalization(momentum=0.9)(a)

    
    y = MaxPooling2D()(a)

    b = Conv2D(128, (3, 3), activation="relu", padding="same")(y)
    b = BatchNormalization(momentum=0.9)(b)

    b = Conv2D(128, (3, 3), activation="relu", padding="same")(b)
    b = BatchNormalization(momentum=0.9)(b)

    
    y = keras.layers.MaxPooling2D()(b)

    c = Conv2D(256, (3, 3), activation="relu", padding="same")(y)
    c = BatchNormalization(momentum=0.9)(c)

    c = Conv2D(256, (3, 3), activation="relu", padding="same")(c)
    c = BatchNormalization(momentum=0.9)(c)

    
    y = MaxPooling2D()(c)

    d = Conv2D(512, (3, 3), activation="relu", padding="same")(y)
    d = BatchNormalization(momentum=0.9)(d)

    d = Conv2D(512,(3, 3), activation="relu", padding="same")(d)
    d = BatchNormalization(momentum=0.9)(d)

    
    # UP

    d = UpSampling2D()(d)

    y = Concatenate(axis=3)([d, c])#, axis=3)

    e = Conv2D(256, (3, 3), activation="relu", padding="same")(y)
    e = BatchNormalization(momentum=0.9)(e)

    e = Conv2D(256,(3, 3), activation="relu", padding="same")(e)
    e = BatchNormalization(momentum=0.9)(e)

    e = UpSampling2D()(e)

    
    y = Concatenate(axis=3)([e, b])#, axis=3)

    f = Conv2D(128, (3, 3), activation="relu", padding="same")(y)
    f = BatchNormalization(momentum=0.9)(f)

    f = Conv2D(128, (3, 3), activation="relu", padding="same")(f)
    f = BatchNormalization(momentum=0.9)(f)

    f = UpSampling2D()(f)

    
    y = Concatenate(axis=3)([f, a])#, axis=3)

    y = Conv2D(64, (3, 3), activation="relu", padding="same")(y)
    y = BatchNormalization(momentum=0.9)(y)

    y = Conv2D(64, (3, 3), activation="relu", padding="same")(y)
    y = BatchNormalization(momentum=0.9)(y)

    return [x, y]

''' 

def get_model_3_class(dim1, dim2, input_dim = 1, activation="softmax"):
    
    [x, y] = get_core(dim1, dim2, input_dim)

   # y = Conv2D(3, (1, 1), activation="relu", padding="same")(y)
    y  = keras.layers.Convolution2D(3,1,1,**option_dict_conv)(y)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)
    
    return model
