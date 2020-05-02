#CONVERT IMAGES

###Script

This script requires C01 image files, a format native to CellProfiler.  
This script utilises ImageMagick's convert command to convert a file format.  
This means if ImageMagick is not installed(as will be the case in most non-unix
systems), it will have to be installed. However, if running this on LUNARC, there
should be no errors in regards to this. In addition to ImageMagick, you should 
also have the Bio-Formats library installed for processing microscopy images. 
More information on this is in the notebook mentioned at the end.

###Running SearchCrop

Lines 2,17, and 27 contain strings for file directories which you will need to 
modify for your own purposes. This is easy to do. Once you change these lines, 
you can simply run it using  

$python3 C01toPNG.py  

Remember, the convert command can be used for various formats.  
Salma and I used it to convert images to 8-bit PNG files, which preserved much 
of the image quality, yet kept a low file size. TIFF is also an acceptable format.
If a different format is desired, lines 32 and 33 are the most important to modify.


###Output

The script takes C01 images and converts them to 8-bit PNG images.  
This can be modified so that they are converted to other image formats as well.

###Notebook

FCellProfiler2Python.html is an html snapshot of a Jupyter notebook written by
Nikolay Oskolkov of NBIS. Here, a step-by-step explanation of how to convert C01 
files, reading images using Python, and superimposing the three channels is 
included. These explanations should be helpful in not just converting the images, 
but other parts of the image processing.
