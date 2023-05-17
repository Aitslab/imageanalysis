import tarfile
import time 
import glob
import os
import bftools
import subprocess
os.chdir('/proj/berzelius-2021-21/users/klara/Segmentation/raw_data/')
#os.makedirs('MFGTMPcx7_170525010001',exist_ok=False)
with tarfile.open("/proj/berzelius-2021-21/users/klara/Segmentation/raw_data/MFGTMPcx7_170526090001.tar.gz") as file:
    my_method = file.getnames()
    for each in my_method:
        #specify suffix of files to be extracted
        if each.endswith('d0.C01'):
            start_1 = time.time()
            file.extract(each)
            
            print(each)
            #end timer test indentation?
            end_1 = time.time()
            print ('extraction time: ', end_1 - start_1)
            png_d0 = each.replace('.C01','.png')
            print(png_d0)
            #what is i j
            start_2 = time.time() 
            subprocess.run(['/proj/berzelius-2021-21/users/Salma-files/envs/bftools/bftools/bfconvert', '-overwrite', '-nogroup', each, png_d0])
           # subprocess.run(['convert', png_d0, '-auto-level', '-depth', '8', '-define', 'quantum:format=unsigned', '-type', 'grayscale', png_d0])
        #end timer 2
            end_2 = time.time()
            print ('conversion time: ', end_2 - start_2)    
       
        #end timer
    #end_1 = time.time()
#print ('time: ', end_1 - start_1)
