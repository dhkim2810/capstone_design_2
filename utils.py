import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import logging
from config import get_default_convnet_setting
from networks import MLP, ConvNet, LeNet, AlexNet, AlexCifarNet, VGG11BN, VGG11, ResNet18, ResNet18BN, ResNet18BN_AP


def get_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--iter_eval', type=int, default=100, help='Evaluate synthetic set based on given iteration period')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='results', help='path to save results')
    parser.add_argument('--loss', type=str, default='GM', help='Gradient Matching(GM) / Attention Loss(AT)', choices=['GM','AT'])
    parser.add_argument('--loss_lambda',type=float, default=0.7, help='Balance factor of GM and AT loss (Only for GM_AT)')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    
    # parser.add_argument('--gpu_id', type=int, default=2, help='GPU Uitilization')
    parser.add_argument('--mask', action='store_true', default=False, help='Add attention mask into dataset')
    parser.add_argument('--mask_prob', type=float, default=1.0, help='Probability of using Attention Mask')
    parser.add_argument('--visualize', action='store_true', default=False, help='Plot training graph for real/synthetic images')

    return parser

def get_path(args):
    # experiment name
    experiment_name = f'{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc'
    if args.mask:
        experiment_name += '_mask'
    
    # trial numbering
    trial = 1
    for path in os.listdir(args.save_path):
        if experiment_name in path:
            trial += 1
    experiment_name += f"_{trial}"
    
    save_path = os.path.join(args.save_path, experiment_name)
    
    # generate folder
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    return save_path


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool

def get_eval_config(args):
    if args.method == 'DSA':
        args.epoch_eval_train = 1000
        args.dc_aug_param = None
    else:
        args.dc_aug_param = None
    
    if args.method == 'DSA':
        args.epoch_eval_train = 1000
    else:
        args.epoch_eval_train = 300