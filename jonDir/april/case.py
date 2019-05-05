import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
##
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
##

aug = glob.glob('C:/Users/jxncx/Documents/BINP37/NOTEBOOK/CASE/tiffAug/*tiff')
xImage = np.zeros(shape = (0,1,100,100))
yImage = []

for filename in aug:
    img = Image.open((filename))
    imgArray = np.array(img)
    imgArray = np.expand_dims(np.expand_dims(imgArray, axis = 0),
                              axis = 0)
    xImage = np.concatenate((xImage, imgArray), axis = 0)
    if 'f00' in filename:
        yImage.append(0)
    else:
        yImage.append(1)
print(xImage.shape)
yImage = np.array(yImage)
print(yImage.shape)
print(yImage[0:10])

xImageTrain, xImageTest, yImageTrain, yImageTest = train_test_split(xImage,
                                                                    yImage,
                                                                    test_size = 0.2)
print("The dimensions of xImageTrain and yImageTrain are",
      xImageTrain.shape, " and ", yImageTrain.shape, " respectively")
print("The dimensions of xImageTest and yImageTest are",
      xImageTest.shape, " and ", yImageTest.shape, " respectively")
##
K.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)
xImageTrain = xImageTrain.astype('float32') / 255.0
xImageTest = xImageTest.astype('float32') / 255.0
yImageTrain = np_utils.to_categorical(yImageTrain)
yImageTest = np_utils.to_categorical(yImageTest)
numClasses = yImageTest.shape[1]

#Model
model = Sequential()
model.add(Conv2D(100, kernel_size=(3, 3), input_shape=(1, 100, 100), padding='same', activation='relu', 
                 kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu', kernel_constraint = maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation = 'sigmoid'))
# Compile model
epochs = 5
lrate = 0.01
decay = lrate / epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
print(model.summary())
estimator = model.fit(xImageTrain, yImageTrain, validation_split = 0.2, shuffle = True, 
                      epochs = epochs, batch_size = 5)
print("Training Accuracy: ",estimator.history['acc'][-1])
print("Validation Accuracy: ",estimator.history['val_acc'][-1])