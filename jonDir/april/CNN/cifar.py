import numpy as np
from keras.datasets import cifar10
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
from matplotlib import pyplot as plt
from scipy.misc import toimage

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

for i in range(0,9):
    plt.subplot(330 + 1 + i)
    plt.imshow(toimage(xTrain[i]))
plt.show()

K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain = xTrain / 255.0
xTest = xTest / 255.0

yTrain = np_utils.to_categorical(yTrain)
yTest = np_utils.to_categorical(yTest)
numClasses = yTest.shape[1]

#Model time
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (3,32,32), padding = 'same',
                 activation = 'relu', kernel_constraint = maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same',
                 kernel_constraint = maxnorm(3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu', kernel_constraint = maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation = 'softmax'))

#Compile time
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay,
          nesterov = False)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd,
              metrics = ['accuracy'])
print(model.summary())

#Fitting time
model.fit(xTrain, yTrain, validation_data = (xTest, yTest), epochs = epochs,
          batch_size = 32)
scores = model.evaluate(xTest, yTest, verbose = 0)
print("Accuracy : %.2f%%" (scores[1] * 100))