#!/usr/bin/env bash
#SBATCH --gpus 2
#SBATCH -A berzelius-2022-57
#SBATCH -t 10:00:00
#SBATCH -n 5
rm -r logs/
python train.py --gpu='0,1'
