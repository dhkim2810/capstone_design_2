#!/bin/bash

#SBATCH --job-name dc_batch_sampling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH --nodelist=sw2
#SBATCH -o logs/slurm-%A-%x.out

python main.py --dataset CIFAR10  --model Conv_Net  --ipc 10 \
    --data_path /local_datasets/ \
    --save_path /data/dhkim2810/capstone/capstone_design_2/results
python main.py --dataset CIFAR10  --model ConvNet  --ipc 10  --init real  --method DSA  \
    --dsa_strategy color_crop_cutout_flip_scale_rotate \
    --data_path /local_datasets/ \
    --save_path /data/dhkim2810/capstone/capstone_design_2/results
# letting slurm know this code finished without any problem
exit 0
