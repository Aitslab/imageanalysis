#Load packages
import argparse
import os
import cv2
import glob
import pandas as pd
#argparse stuff
parser = argparse.ArgumentParser()
parser.add_argument('c', type = str, help = 'cell profiler data')
parser.add_argument('im', type = str, help = 'directory with microscopy images')
args = parser.parse_args()
#####################

#Import and process cellPro output data
os.chdir('/lunarc/nobackup/users/jlis/bigCrop')
print("Directory changed to script directory // READING CSV DATA")
imData = pd.read_csv(args.c)
print("DATA DONE READING")
imData = imData.drop(imData.columns.difference(['ImageNumber', 
                                                        'ObjectNumber',
                                                        'Metadata_FileLocation',
                                                        'Metadata_Plate',
                                                        'Metadata_Well',
                                                        'Location_Center_X',
                                                        'Location_Center_Y']),
                           axis = 1)
imData.to_csv("bigObjects.csv", index = False)
print("DATA SAVED to bigObjects.csv")
#Import and process image(s)
imRaw = glob.glob(args.im + '/*')
imTest = cv2.imread(imRaw[0])
imExt = imRaw[0].split('.')[-1]
dim = imTest[0].shape
counter = 1

#Using set to get a list of all image names
imgSet = set(imData['Metadata_FileLocation'])
subdirs = ['ch0Res', 'ch1Res', 'ch2Res']
for subdir in subdirs:
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    else:
        os.system('rm -rf %s/*' % subdir)
for i in imgSet:
	imName = i.split('/')[-1][:-4]
	imCh = imName[-2:]
	if imCh == 'd0':
		subdir = subdirs[0]
	elif imCh == 'd1':
		subdir = subdirs[1]
	elif imCh == 'd2':
		subdir = subdirs[2]
	try:
		nonBordIm = imData.loc[(imData['Metadata_FileLocation'] == i) &
				(imData['Location_Center_X'] > 200) &
				(imData['Location_Center_X'] < (dim[0] - 200)) &
				(imData['Location_Center_Y'] > 200) &
				(imData['Location_Center_Y'] < (dim[0] - 200))].iloc[0]
		x = int(nonBordIm['Location_Center_X'])
		y = int(nonBordIm['Location_Center_Y'])
		objNum = str(nonBordIm['ObjectNumber'])
		#print("READING IMAGE {0} NAMED {1}".format(counter, imName))
		imageVar = cv2.imread(args.im + '.' + imExt)
		#print("CROPPING IMAGE {0} NAMED {1}".format(counter, imName))
		cropImg = imageVar[y - 200:y + 200, x - 200:x + 200].copy()
		cv2.imwrite("searchCrop/{0}/{1}_{2}_{3}_{4}.{5}".format(subdir,
			imName, objNum, str(x), str(y), imExt), cropImg)
		print("IMAGE SAVED FOR --- {0} \n PROGRESS : {1}".format(imName,
			((counter / len(imgSet)) * 100)))
		counter += 1
	except IndexError:
		print("SOMETHING WRONG WITH IMAGE, MOVING ON")
		counter += 1
print("CROPPING FOR {} IMAGES COMPLETE".format(len(imgSet)))
		
