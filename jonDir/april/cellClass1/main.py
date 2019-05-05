from PIL import Image
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
import glob
import numpy as np
fileNames = glob.glob('C:/Users/jxncx/Documents/BINP37/NOTEBOOK/CELLCLASS1/tiffDownIm/*.tiff')
images = [Image.open(i) for i in fileNames]
imgnp = [np.array(i) for i in images]
imgnp = [np.expand_dims(np.expand_dims(i, axis = 0)
        , axis = 0) for i in imgnp]

trainValue = 294
testValue = 126

xTrain = np.concatenate((imgnp[0:trainValue]), axis = 0)
xTest = np.concatenate((imgnp[trainValue:]), axis = 0)
yTrain = [0 for i in range(int(trainValue/2))]
secondClass = [1 for i in range(int(trainValue/2))]
yTrain.extend(secondClass)
yTrain = np.array(yTrain)
yTest = [0 for i in range(int(testValue/2))]
secondClass = [1 for i in range(int(testValue/2))]
yTest.extend(secondClass)
yTest = np.array(yTest)

K.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)

xTrain = xTrain.astype('float32') / 255.0
xTest = xTest.astype('float32') / 255.0
xTrain.reshape(trainValue,100,100,1)
xTest.reshape(testValue,100,100,1)
yTrain = np_utils.to_categorical(yTrain)
yTest = np_utils.to_categorical(yTest)
numClasses = yTest.shape[1]

#Model time
model = Sequential()

model.add(Conv2D(100, (3, 3), input_shape=(1, 100, 100), padding='same', activation='relu', 
                 kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation='sigmoid'))

epochs = 15
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
model.compile(loss = 'binary_crossentropy', optimizer = sgd,
              metrics = ['accuracy'])
print(model.summary())

#Training the model
model.fit(xTrain, yTrain, validation_data = (xTest, yTest),
          epochs = epochs, batch_size = 1)

#final eval
scores = model.evaluate(xTest, yTest, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100))