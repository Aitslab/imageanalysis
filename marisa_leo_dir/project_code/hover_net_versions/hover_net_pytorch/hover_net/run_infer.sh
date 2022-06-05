#!/usr/bin/env bash
#SBATCH --gpus 4
#SBATCH -A berzelius-2022-57
#SBATCH -t 1:00:00
#SBATCH -n 5
./run_tile.sh
