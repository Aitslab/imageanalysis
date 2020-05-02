#SEARCH AND CROP SCRIPT

###CellProfiler

This script requires a data table from CellProfiler.  
Familiarise yourself with using CellProfiler.  
The pipeline to use is already included in this directory.  
Import the images, import the pipeline, and change the output directories.  
Run the analysis and find the resulting CSV.  
It should have coordinates of objects found.  

###Running SearchCrop

SearchCrop takes two arguments.  
The first positional argument is the CellProfiler data in the form of a csv.  
The second positional argument is the directory with the images.  
With the coordinates from the csv, the script can find the object in the images.  
I quickly modified the script so it would work better in general cases, so it  
might not work at first when you try it. If this is the case, please take a  
look at the code or email jo3500li-s@student.lu.se  

###Output

The script creates directories for each of the 3 channels, if they don't exist.  
If they do exist, it just removes all the contents of the directories.  
This is built to run on the LUNARC server which uses bash commands.   
If you try to run this on a Windows system, it will not work as it uses the rm
command.  
Then, the 400x400 cropped images and saved in the subdirectory corresponding to
its channel. If the image is corrupted, there is a try/except block, as they
cannot be cropped without a struggle.  

###Notebook

Oct.ipynb is a Jupyter notebook which can be opened using Jupyter lab/notebook.  
Read it to see more in detail how the code works. This can be useful for
troubleshooting any issues.
