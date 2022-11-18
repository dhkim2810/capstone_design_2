#!/bin/bash

#SBATCH --job-name dc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 3-0
#SBATCH --partition batch_ce_ugrad
#SBATCH --nodelist=sw1
#SBATCH -o logs/slurm-%A-%x.out

for ipc in 1 10 50
do
    for layer in 1 2 3
    do
        python main_BS.py --dataset CIFAR10 --model ConvNet  --ipc $ipc \
            --data_path /local_datasets/ --layer_idx $layer \
            --save_path /data/dhkim2810/capstone/capstone_design_2/results
    done
done

for ipc in 1 10 50
do
    for layer in 1 2 3
    do
        python main_BS.py --dataset CIFAR10 --model ConvNet  --ipc $ipc  --init real  --method DSA \
            --dsa_strategy color_crop_cutout_flip_scale_rotate \
            --data_path /local_datasets/ --layer_idx $layer \
            --save_path /data/dhkim2810/capstone/capstone_design_2/results
    done
done
# letting slurm know this code finished without any problem
exit 0
