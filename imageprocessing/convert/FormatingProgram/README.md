# Change image format

This script will allow you to change format of all images in one folder to one of following formats: C01, png, tiff, jpg.

## Usage
```bash
python3 format_conversion.py -i INDIR -o OUTDIR -ift IN_FILETYPE -oft OUT_FILETYPE
```
Choose a directory with your files to convert, which format they are (e.g. C01, png, tiff, jpg), and name the output directory where you want to place the converted images, and what format you want to convert them to (e.g. C01, png, tiff, jpg).



## Installation

### Installation using conda

If you have [conda](https://docs.anaconda.com/anaconda/install/), you can set up an environment as follows:

```bash
conda create -n imgConv python=3.6
conda activate imgConv
```

And install following:

bftools
```
conda install -c bioconda bftools
```

And non standard python libraries:
```bash
conda install tqdm
```

If you get a java error, you might need to install it:
```bash
conda install -c bioconda java-jdk
```


### Regular installation

Download bftools and unzip it to your /home/bin folder and add to path:

```bash
cd ~/bin
wget http://downloads.openmicroscopy.org/latest/bio-formats/artifacts/bftools.zip
unzip bftools.zip
rm bftools.zip
export PATH=$PATH:~/bin/bftools
```
Download and install java:

Download from https://www.java.com/en/download/linux_manual.jsp

(following commands are for the download for Linux x64)

```bash
cd ~/Downloads
tar -C ~/bin -zxvf jre-8u261-linux-x64.tar.gz
rm jre-8u261-linux-x64.tar.gz 
export PATH=$PATH:~/bin/jre-8u261-linux-x64/bin
```

Required non standard python libraries
```python
pip install tqdm
```

