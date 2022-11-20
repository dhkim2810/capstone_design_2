#!/bin/bash

#SBATCH --job-name dc_batch_sampling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH --nodelist=sw1
#SBATCH -o logs/slurm-%A-%x.out

layer_idx = 2

for dataset in CIFAR10 CIFAR100 SVHN
do
    for ipc in 1 10 50
    do
        python main_BS.py --dataset CIFAR10 --model ConvNet  --ipc $ipc \
            --data_path /local_datasets/ --layer_idx $layer_idx \
            --save_path /data/dhkim2810/capstone/capstone_design_2/results
        python main_BS.py --dataset CIFAR10 --model ConvNet  --ipc 50  --init real  --method DSA  \
            --dsa_strategy color_crop_cutout_flip_scale_rotate \
            --data_path /local_datasets/ --layer_idx $layer_idx \
            --save_path /data/dhkim2810/capstone/capstone_design_2/results
    done
done

# letting slurm know this code finished without any problem
exit 0
