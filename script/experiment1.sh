#!/bin/bash
# Testing Baseline Accuracy

for ipc in 10 1 50
do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --model ConvNet \
        --ipc $ipc \
        --data_path ./datasets \
        --save_path results/exp1 \
        --cluster_path clustering
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --model ConvNet  --init real  --method DSA  \
        --ipc $ipc \
        --data_path ./datasets --dsa_strategy color_crop_cutout_flip_scale_rotate \
        --save_path results/exp1 \
        --cluster_path clustering
    CUDA_VISIBLE_DEVICES=1 python main_BS.py --dataset CIFAR10 --model ConvNet \
            --num_cluster 20 --ipc $ipc --layer_idx 2 --norm \
            --data_path ./datasets \
            --save_path results/exp1 \
            --cluster_path clustering
    CUDA_VISIBLE_DEVICES=1 python main_BS.py --dataset CIFAR10 --model ConvNet  --init real  --method DSA  \
        --num_cluster 20 --ipc $ipc --layer_idx 2 --norm
        --data_path ./datasets --dsa_strategy color_crop_cutout_flip_scale_rotate \
        --save_path results/exp1 \
        --cluster_path clustering
done