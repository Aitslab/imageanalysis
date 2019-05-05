import glob #10
import numpy as np #38,39,40,60,85,87,95
import matplotlib.pyplot as plt #27:37,43:46,51,52,56,57,65,69,70,73,104:115
import os #82,92
from matplotlib import image #55,68
from skimage import io 
from PIL import Image #23,24,25,42,50,61,84,94,96,97,122
from os import listdir #66,81

tiff = glob.glob('C:/Users/jxncx/Documents/BINP37/CP/qgsd/set1/tiff_images/*.tiff')

fig = plt.figure(figsize = (20,15))
for i,j in zip(tiff, range(len(tiff))):
    img = io.imread(i)
    plt.subplot(231 + j)
    plt.imshow(img[:,:], cmap = plt.cm.gray)
    plt.title(i.split('/')[8].split('.')[0], fontsize = 20)

img = io.imread('C:/Users/jxncx/Documents/BINP37/CP/qgsd/set1/tiff_images/MFGTMPcx7_170621060001_A01f10d0.tiff')
plt.tight_layout()
plt.show()

im1 = Image.open('tiff_images/MFGTMPcx7_170621060001_A01f07d0.tiff')
im2 = Image.open('tiff_images/MFGTMPcx7_170621060001_A01f07d1.tiff')
im3 = Image.open('tiff_images/MFGTMPcx7_170621060001_A01f07d2.tiff')

fig = plt.figure(figsize = (20,15))
plt.figure(figsize = (12,8))
plt.subplot(131)
plt.imshow(im1)
plt.subplot(132)
plt.imshow(im2)
plt.subplot(133)
plt.imshow(im3)
plt.tight_layout()
plt.show()

imarray1 = np.array(im1)
imarray2 = np.array(im2)
imarray3 = np.array(im3)

newImage = Image.merge('RGB', [im1, im2, im3])
plt.figure(figsize = (12,8))
plt.imshow(newImage)
plt.show()
#Image Processing
#We will perform image processing using Pillow Python module.

#Pillow...
testImage = Image.open('tiff_images/MFGTMPcx7_170621060001_A01f00d0.tiff')
plt.imshow(testImage)
plt.show()

#mit matplotlib
data = image.imread('tiff_images/MFGTMPcx7_170621060001_A01f00d0.tiff')
plt.imshow(data)
plt.show()

#Access pixel data using numpy
data = np.asarray(testImage)
image2 = Image.fromarray(data)

#Load all images in a directory
loaded = []
fig = plt.figure(figsize = (20,15))
tiffDir = listdir('tiff_images')
for filename, j in zip(tiffDir, range(len(tiffDir))):
    imgData = image.imread('tiff_images/' + filename)
    plt.subplot(231 + j)
    plt.imshow(imgData)
    loaded.append(imgData)
    print('> loaded %s %s' % (filename, imgData.shape))
plt.show()
#For some reason this produces different output than
#Nikolai's notebook, also produces a ValueError num must be 1...
#Which has to do with matplotlib's subplot method

#Image downsizing, images get reduced to 100x100
#File size is about 1/100th of the original
#Cool
tiffDir = listdir('tiff_images')
tiffDown = os.mkdir('tiffDownIm')
for filename in tiffDir:
    imgData = Image.open('tiff_images/' + filename)
    print('> loaded %s %s' % (filename, np.array(imgData).shape))
    imgData.thumbnail((100,100))
    print('> downsized image: {}'.format(np.array(imgData).shape))
    print('> saving downsized image {}'.format(filename))
    imgData.save('tiffDownIm/' + filename)

#Creating an augmented data set
tiffAug = os.mkdir('tiffAug')
for filename in tiffDir:
    imgData = Image.open('tiffDownIm/' + filename)
    print('> loaded %s %s' % (filename, np.array(imgData).shape))
    hFlip = imgData.transpose(Image.FLIP_LEFT_RIGHT)
    vFlip = imgData.transpose(Image.FLIP_TOP_BOTTOM)
    print('> saving augmented images for {}'.format(filename))
    imgData.save('tiffAug/' + filename)
    hFlip.save('tiffAug/augHorziontal' + filename)
    vFlip.save('tiffAug/augVertical' + filename)

#Rotated versions of images: 45, 90, etc...
fig = plt.figure(figsize = (20,15))
plt.subplot(131)
plt.title("Original Image", fontsize = 20)
plt.imshow(testImage)
plt.subplot(132)
plt.imshow(testImage.rotate(45))
plt.title("Rotate 45 Degrees", fontsize = 20)
plt.subplot(133)
plt.imshow(testImage.rotate(90))
plt.title("Rotate 90 degrees", fontsize = 20)
plt.show()

#Augmented data set with rotating each image from
#0 to 360 degrees by 2 degree step
counter = 0
for filename in tiffDir:
    print('Augmenting image {}'.format(filename))
    for i in range(0,360,2):
        imgData = Image.open('tiffDownIm/' + filename)
        rotImage = imgData.rotate(i)
        rotImage.save('tiffAug/aug_' + str(i) + '_' + filename)
    counter += 1
    print(str((counter/(len(tiffDir))) * 100) + "% Complete")