import subprocess
import glob
c01Files = glob.glob('C:/Users/jxncx/Documents/BINP37/CP/qgsd/set1/co/*.C01')
tiffy = [str(i.split('.C01')[0]) + '.tiff' for i in c01Files]
tiffy = [i.replace('co', 'tiff') for i in tiffy]

counter = 0
for i,j in zip(c01Files, tiffy):
    counter += 1
    print('Converting {} to tiff format'.format(i))
    subprocess.run(['C:/Users/jxncx/Documents/BINP37/CP/qgsd/set1/bftools/bfconvert.bat',
                    '-overwrite', '-nogroup', i, j])
    print('{} images converted'.format(counter))
print('FERTIG')