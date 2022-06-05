#!/usr/bin/env bash
#SBATCH --gpus 4
#SBATCH -A berzelius-2022-57
#SBATCH -t 3:00:00
#SBATCH -n 5
#python run_train.py --gpu='0,1,2,3' > cell_results.txt
./epoch_run.sh
