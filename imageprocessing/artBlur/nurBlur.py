import random #Randomising dead pixel location
import os #Changing working directory
import glob #File and directory handling
import cv2 #For image manipulating functions
#####################
#argparse stuff
parser = argparse.ArgumentParser()
parser.add_argument('i', type = str, help = 'input directory')
parser.add_argument('o', type = str, help = 'output directory')
args = parser.parse_args()
#####################
#FX : Namer
def namer(image):
    if '\\' in image:
        imgName = image.split('\\')[-1]
    elif '/' in image:
        imgName = image.split('/')[-1]
    return imgName

#FX : Blurs images with degrees of varying averaging
def blur(image, out):
    imgName = namer(image)
    fileExtSplit = imgName.split('.')
    klist = [(5,5), (10,10), (25,25), (40,40), (50,50)]
    kparam = random.choice(klist)
    img = cv2.imread(image)
    blurred = cv2.blur(img, kparam)
    cv2.imwrite("{0}\{1}_{2}.{3}".format(out, fileExtSplit[0], kparam[0],
                                        fileExtSplit[1]), blurred)
    
#FX : Main function
def main():
    inDir = glob.glob(args.i + '/*')
    outDir = glob.glob(args.o + '/*')
    if outDir:
        for image in outDir:
            os.remove(image)
    counter = 0
    corr = 0
    for file in inDir:
        try:
            blur(file)
        except:
            corr += 1
            print("CORRUPTED IMAGE --- COULD NOT BLUR")
        finally:
            counter += 1
            print("3. BLURRING IMAGES --- [{0} / {1} --- {2} %]".format(counter, len(inDir),
                                                                    ((counter / len(inDir)) * 100)))
            print("TOTAL CORRUPTED IMAGES ENCOUNTERED : {0}".format(corr))
#Run
if __name__ == "__main__":
    main()