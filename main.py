import os
import copy
import wandb
import numpy as np
import argparse

import torch

from data import AugmentModule, DataModule
from model import ModelModule
import utils

def main(args):
    # Basic Setup
    np.random.seed(args.config.seed)
    
    ## Directory Setup
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    ## Logging Setup
    # run_name = '-'.join(args.method, args.dataset, f'{args.ipc}ipc', args.arch, args.comment)
    # wandb.init(dir=args.save_dir, config=args, entity="dhk", project="capstone2",
    #            tags=[args.dataset, args.arch, args.ipc], name=run_name,)
    
    ## CUDA Init
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device_count = torch.cuda.device_count()
    print(f"Using CUDA : Device count : {args.device_count}")
    
    ## Dataset Config
    dm = DataModule(args.data_dir, args.dataset)
    dm_config = dm.get_dataset_config()
    
    ## Synthetic Dataset Config
    image_syn = torch.randn(size=(dm_config['num_classes'] * args.ipc, dm_config['channel'], dm_config['im_size'][0], dm_config['im_size'][1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.cat([torch.ones(1,args.ipc, dtype=torch.long, device=args.device)*i for i in range(dm_config['num_classes'])])
    optimizer_img = torch.optim.SGD([image_syn,], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()
    
    ## Training Configuration
    args.outer_loop, args.inner_loop = utils.get_loops()
    eval_iter_pool = np.arange(0,args.epochs,args.iter_eval).tolist() if args.eval_mode == 'S' else [args.epochs]
    model_eval_pool = utils.get_eval_pool(args.eval_mode, args.arch)
    
    ## Start Training
    for iter in range(args.epochs+1):
        ''' Evaluate Synthetic Dataset '''
        print("Eval Synthetic Dataset")
        if iter in eval_iter_pool:
            for model_eval in model_eval_pool:
                epoch_eval_train = utils.get_eval_epoch(args)
                aug_eval = AugmentModule(args, dm_config['im_size'])
                for iter_eval in range(args.eval_num):
                    print(f"Eval {iter_eval}")
                    net_eval = ModelModule(model_eval, aug_eval, dm_config, args.__dict__)
                    eval_syn_images, eval_syn_labels = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                    _, _ = net_eval.train_with_synthetic_data(eval_syn_images, eval_syn_labels, epoch_eval_train, lr_schedule=True)
                    eval_test_loss, eval_test_acc = net_eval.test_with_synthetic_data(eval_syn_images, eval_syn_labels)
        
        '''  Update Synthetic Dataset  '''
        print("Update Synthetic Dataset")
        ## Augment
        augment = AugmentModule(args, dm_config['im_size'])
        
        ## Model
        model = ModelModule(args.model, augment, dm_config, args.__dict__)
        
        print("Start training")
        ## Outer Loop
        log_matching_loss = 0.0
        for ol in range(args.outer_loop):
            print("Freeze BN")
            ''' freeze the running MU and sigma for BN layers '''
            BNSizePC = 32
            img_real = torch.cat([dm.get_real_images(class_idx, BNSizePC) for class_idx in range(dm_config['num_classes'])], dim=0).to(args.device)
            freeze = model.freeze_model_BN_layers(img_real)
            
            print("Update Dataset")
            ''' update synthetic dataset '''
            matching_loss = torch.tensor(0.0).to(args.device)
            for class_idx in range(dm_config['num_classes']):
                img_real = dm.get_real_images(class_idx, args.batch_real).to(args.device)
                label_real = class_idx * torch.ones((img_real.shape[0],), dtype=torch.long, device=args.device)
                img_syn = image_syn[class_idx*args.ipc:(class_idx+1)*args.ipc].reshape((args.ipc, dm_config['channel'], dm_config['im_size'][0],dm_config['im_size'][1]))
                label_syn = class_idx * torch.ones((img_real.shape[0],), dtype=torch.long, device=args.device)
                
                matching_loss += model.calculate_matching_loss(img_real, label_real, img_syn, label_syn)
            optimizer_img.zero_grad()
            matching_loss.backward()
            optimizer_img.step()
            log_matching_loss += matching_loss.item()
            
            print("Update Model")
            ''' update network '''
            train_syn_images, train_syn_labels = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
            train_syn_losses, train_syn_accs = model.train_with_synthetic_data(train_syn_images, train_syn_labels, epochs=args.inner_loop, batch_size=args.batch_size_train, freeze=freeze)
        log_matching_loss /= args.outer_loop
        
        # Log Data
        # wandb.log({
        #     'Matching Loss' : log_matching_loss,
        #     'Syn Train Loss' : train_syn_losses[-1],
        #     'Syn Test Loss' : None,
        #     'Syn Train Accuracy' : train_syn_accs[-1],
        #     'Syn Test Accuracy' : None,
        # })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--log_dir', type=str, default='./log', help='dataset path')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='dataset path')
    parser.add_argument('--save_dir', type=str, default='./results', help='path to save results')
    
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA/Ours', choices=['DC','DSA','Ours'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset', choices=['MNIST','FashionMNIST','SVHN','CIFAR10','CIFAR100'])
    parser.add_argument('--arch', type=str, default='ConvNet', help='model used for condensation', choices=['AlexNet, ConvNet, ResNet18, ResNet18BN, ResNet18AP_BN'])
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    
    parser.add_argument('--epochs', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_img_schedule', type=int, default=[1200,1400,1800], nargs='+')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for real data')
    
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode', choices=['S','M'])
    parser.add_argument('--iter_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--num_eval', type=int, default=10, help='number of evaluation')
    
    parser.add_argument('--strategy', type=str, default='none', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--noise', type=float, default=0.001, help='noise factor for noise augmentation')
    parser.add_argument('--rotate', type=int, default=45, help='angle for rotation augmentation')
    parser.add_argument('--scale', type=float, default=0.2, help='scale factor for scale augmentation')
    
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dis_metric', type=str, default='dc', help='distance metric')
    parser.add_argument('--comments', type=str, help='comments for run name')

    args = parser.parse_args()
    main(args)