#!/bin/sh
for VARIABLE in `seq 1 50`
do
python run_infer.py --model_path="logs/00/net_epoch=${VARIABLE}.tar" --model_mode='original' tile --input_dir='../dataset/Test/Images' --output_dir="output/${VARIABLE}"

done
