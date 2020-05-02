import os
os.chdir('/lunarc/nobackup/users/salka/Downloads/next_folders/')
# upper directory of image folders

# BUILD LIST OF PLATES
plate_list = list()
for i in os.listdir():
    print(i)
    if os.path.isdir(i):
        plate_list.append(i)
        print(plate_list)
print('There are {} plates\n'.format(len(plate_list)))
## I had just one folder
import glob
C01_files = list()
for i in plate_list:
    C01_files_plate = glob.glob('/lunarc/nobackup/users/salka/Downloads/next_folders/Saved_selected_images/' + \
                                str(i) + '/*.C01')
    print('Plate {0} has {1} C01-files'.format(i, len(C01_files_plate)))
    C01_files = C01_files + C01_files_plate
print('\n')
print('In total there are {} C01-files on all plates\n'.format(len(C01_files)))
print(C01_files[0:10])
print('\n')
# MAKE FOLDERS FOR png-FILES
for i in range(len(plate_list)):
    if not os.path.exists('/lunarc/nobackup/users/salka/Downloads/next_folders/Saved_png_images/' + plate_list[i]):
        print('Making directory {}'.format(plate_list[i]))
        os.mkdir('/lunarc/nobackup/users/salka/Downloads/next_folders/Saved_png_images/' + plate_list[i])

# CREATE LIST OF NAMES FOR png-FILES
png_files = [str(i.split('.C01')[0]) + '.png' for i in C01_files]
png_files = [i.replace('images','png_images') for i in png_files]
print(png_files[0:10])
print('\n')

# CONVERT C01 TO 8-BIT png FORMAT
import subprocess
k = 0
for i,j in zip(C01_files, png_files):
    if os.path.isfile(j):
        print('The png-file {} already exists'.format(j))
        next
    else:
        print('Converting {} to 8-bit png format'.format(i))
        subprocess.run(['bfconvert', '-overwrite', '-nogroup', i, j])
        subprocess.run(['convert', j, '-auto-level', '-depth', '8', '-define', 'quantum:format=unsigned', '-type', 'grayscale', j])
        k = k + 1
        if k%10 == 0:
            print('\nFinished conversion for {} files'.format(k))
            print("********************************************\n")
