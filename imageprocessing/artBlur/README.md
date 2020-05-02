#NURBLUR SCRIPT

###Script

This script blurs images using a function built into OpenCV which uses averaging.  
Very mathematically simple, this just takes the average of a kernel and applies
it to the area over the kernel. The script randomises the size of this kernel.
The larger the kernel, the larger the area being blurred.  
So obviously, a kernel with size (40,40) will have a more severe blur than that
of size (5,5).

###Running SearchCrop

SearchCrop takes two arguments.  
The first positional argument is the directory for the non-blurry input.  
The second positional argument is the directory for the blurry output.  
Simply run it as python3 nurBlur.py -i -o.

###Output

The output is blurry images with random degrees of blurriness. 

###Notebook

Nov.ipynb is a Jupyter notebook which can be opened using Jupyter lab/notebook.  
Read it to see more in detail how the code works. This can be useful for
troubleshooting any issues. This includes explanations and code snippets from
the more overarching script which artificially creates the dead pixel error.
