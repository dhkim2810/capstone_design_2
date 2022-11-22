#!/bin/bash
# Testing ClusterWise Accuracy

for ipc in 10 1 50
do
    for num_cluster in 5 10 15 25
    do
        CUDA_VISIBLE_DEVICES=2 python main_BS.py --dataset CIFAR10 --model ConvNet \
            --num_cluster $num_cluster --ipc $ipc --layer_idx 2 --norm \
            --data_path ./datasets \
            --save_path results/exp3 \
            --cluster_path clustering
    done
done
# letting slurm know this code finished without any problem
exit 0
