#!/bin/bash

#SBATCH --job-name dc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 3-0
#SBATCH --partition batch_ce_ugrad
#SBATCH --nodelist=sw1
#SBATCH -o logs/slurm-%A-%x.out

python main_BS.py --dataset CIFAR10 --model ConvNet  --ipc 10 \
    --data_path /local_datasets/ --layer_idx 2 --norm --num_cluster $num_cluster \
    --save_path /data/dhkim2810/capstone/capstone_design_2/results \
    --cluster_path /data/dhkim2810/capstone/capstone_design_2/clustering \

python main_BS.py --dataset CIFAR10 --model ConvNet  --ipc 10 \
    --data_path /local_datasets/ --layer_idx 2 --num_cluster $num_cluster \
    --save_path /data/dhkim2810/capstone/capstone_design_2/results \
    --cluster_path /data/dhkim2810/capstone/capstone_design_2/clustering \

# letting slurm know this code finished without any problem
exit 0
