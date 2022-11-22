#!/bin/bash

#SBATCH --job-name dc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 3-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -o logs/slurm-%A-%x.out

for layer in 0 1 2 3
do
    python main_BS.py --dataset CIFAR10 --model ConvNet \
        --num_cluster 20 --ipc 10 --layer_idx $layer --norm \
        --data_path /local_datasets/ \
        --save_path /data/dhkim2810/capstone/capstone_design_2/results \
        --cluster_path /data/dhkim2810/capstone/capstone_design_2/clustering
    python main_BS.py --dataset CIFAR10 --model ConvNet --init real --method DSA \
        --num_cluster 20 --ipc 10 --layer_idx $layer --norm \
        --data_path /local_datasets/ --dsa_strategy color_crop_cutout_flip_scale_rotate \
        --save_path /data/dhkim2810/capstone/capstone_design_2/results \
        --cluster_path /data/dhkim2810/capstone/capstone_design_2/clustering
done
# letting slurm know this code finished without any problem
exit 0