#!/usr/bin/python3

import argparse
import os
import os.path
import subprocess
from tqdm import tqdm
from pathlib import Path

#################################### ARGPARSE ##################################
usage = 'Enter the directory name where the files to convert are located, what format to convert the files to \
and name a directory where you want the converted files to end up.'
parser = argparse.ArgumentParser(description=usage)
parser.add_argument(
	'-i',
	dest = 'infile',
	metavar = 'INDIR',
	type = str,
    help = 'set the directory where the input files are located',
	required = True
	)
parser.add_argument(
	'-o',
	dest = 'outfile',
	metavar = 'OUTDIR',
	type = str,
    help = 'set the directory to store the converted files, directory will be created if not already existing.',
	required = True
	)
parser.add_argument(
	'-ift',
	dest = 'input_filetype',
	metavar = 'IN_FILETYPE',
	type = str,
    help = 'Set what format the input files are, e.g C01 png',
	required = True
	)
parser.add_argument(
	'-oft',
	dest = 'output_filetype',
	metavar = 'OUT_FILETYPE',
	type = str,
    help = 'Chose format to convert to, e.g. tiff or png',
	required = True
	)
args = parser.parse_args()
################################################################################

# Convert the input to the absolute path
input_dir = os.path.abspath(args.infile)
output_dir = os.path.abspath(args.outfile)


out_filetype = '.{}'.format(args.output_filetype)
in_filetype = args.input_filetype

# If the output directory does not exist,
# a directory will be created with that name.
my_file = Path(output_dir)
if not my_file.exists():
    os.mkdir(output_dir)

# If the path provided is not a directory, raise error
if not os.path.isdir(input_dir):
    raise argparse.ArgumentTypeError('Input must be a directory')
if not os.path.isdir(output_dir):
    raise argparse.ArgumentTypeError('Output must be a directory')

input_files = []
converted_files = []
os.chdir(input_dir)
for i in os.listdir(input_dir):
    if i.split('.')[-1] == in_filetype: # Checks that filename ends with format chosen
        input_files.append(input_dir + '/' + i)
        converted_files.append(output_dir + '/' + i.split('.')[0] + out_filetype)

for i,j in tqdm(zip(input_files,converted_files), total = len(input_files)): # tqdm creates a progressbar to see the progress.
    subprocess.run(['bfconvert', '-overwrite', '-nogroup',i,j],stdout = subprocess.PIPE, stderr = subprocess.DEVNULL) #Runs bftools which needs to be preinstalled, output to DEVNULL.
    subprocess.run(['convert', i, '-auto-level', '-depth', '16', '-define', 'quantum:format=unsigned', '-type', 'grayscale', j],stdout = subprocess.PIPE, stderr = subprocess.DEVNULL) #Convert images to 16-bits tiff images.
