from nis import match
import os
import time
import copy
import numpy as np
import wandb

import torch
import torch.nn as nn

from data.data_module import DataModule
from data.aug_module import AugmentModule
from model.model_module import ModelModule
import utils

def main(args):
    # Basic Setup
    np.random.seed(0)
    
    # Directory Setup
    
    # Logging Setup
    run_name = '-'.join(args.dataset, f'{args.ipc}ipc', args.arch, args.method, args.comment)
    wandb.init(dir=args.save_dir, config=args, entity="dhk", project="capstone2",
               tags=[args.dataset, args.arch, args.ipc], name=run_name,)
    
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
    eval_iter_pool = np.arange(0,args.epochs,args.iter_eval).tolist() if args.eval_model == 'S' else [args.epochs]
    model_eval_pool = utils.get_eval_pool(args.eval_mode, args.arch, args.arch)
    
    ## Start Training
    for iter in range(args.epochs+1):
        ''' Evaluate Synthetic Dataset '''
        if iter in eval_iter_pool:
            for model_eval in model_eval_pool:
                epoch_eval, augment_eval = utils.get_eval_config(args)
                aug_eval = AugmentModule(augment_eval)
                for iter_eval in range(epoch_eval):
                    net_eval = ModelModule(model_eval, aug_eval, dm_config['channel'], dm_config['num_classes'], dm_config['im_size'], args.__dict__)
                    eval_syn_images, eval_syn_labels = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                    _, _ = net_eval.train_with_synthetic_data(eval_syn_images, eval_syn_labels, epoch_eval, lr_schedule=True)
                    eval_test_loss, eval_test_acc = net_eval.test_with_synthetic_data(eval_syn_images, eval_syn_labels)
        
        
        '''  Update Synthetic Dataset  '''
        ## Model
        model = ModelModule(args.model, dm_config['channel'], dm_config['num_classes'], dm_config['im_size'], args.device)

        ## Outer Loop
        log_matching_loss = 0.0
        for ol in range(args.outer_loop):
            ''' freeze the running MU and sigma for BN layers '''
            BNSizePC = 32
            img_real = torch.cat([dm.get_real_images(class_idx, BNSizePC) for class_idx in range(dm_config['num_classes'])], dim=0).to(args.device)
            freeze = model.freeze_model_BN_layers(img_real, BNSizePC)
            
            ''' update synthetic dataset '''
            matching_loss = torch.tensor(0.0).to(args.device)
            for class_idx in range(dm_config['num_classes']):
                img_real = dm.get_real_images(class_idx, args.batch_real)
                label_real = class_idx * torch.ones((img_real.shape[0],), dtype=torch.long, device=args.device)
                img_syn = image_syn[class_idx*args.ipc:(class_idx+1)*args.ipc].reshape((args.ipc, dm_config['channel'], dm_config['im_size'][0],dm_config['im_size'][1]))
                label_syn = class_idx * torch.ones((img_real.shape[0],), dtype=torch.long, device=args.device)
                
                matching_loss += model.calculate_matching_loss(img_real, label_real, img_syn, label_syn)
            optimizer_img.zero_grad()
            matching_loss.backward()
            optimizer_img.step()
            log_matching_loss += matching_loss.item()
            
            ''' update network '''
            train_syn_images, train_syn_labels = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
            train_syn_losses, train_syn_accs = model.train_with_synthetic_data(train_syn_images, train_syn_labels, epochs=args.inner_loop, batch_size=args.batch_size_train, freeze=freeze)
        log_matching_loss /= args.outer_loop
        
        # Log Data
        wandb.log({
            'Matching Loss' : log_matching_loss,
            'Syn Train Loss' : train_syn_losses[-1],
            'Syn Test Loss' : None,
            'Syn Train Accuracy' : train_syn_accs[-1],
            'Syn Test Accuracy' : None,
        })

# import os
# import sys
# import time
# import copy
# import numpy as np
# from tqdm import tqdm
# import logging
# import wandb

# import torch
# import torch.nn as nn

# from utils import get_arguments, get_network, get_path, match_loss, get_time
# from config import get_loops, get_eval_pool
# from data import get_dataset, get_daparam, TensorDataset
# from diffaugment import DiffAugment, ParamDiffAug
# from process import epoch, evaluate_synset


# def main(args):
#     #### Training Configuration ####
#     ## Directory Init ##
#     args.save_path = get_path(args)

#     ## Logging Init ##
#     logging.basicConfig(filename=os.path.join(args.save_path, 'logging.log'),
#                         filemode='a',
#                         format='[%(asctime)s] %(levelname)s:%(message)s',
#                         datefmt='%m/%d %H:%M:%S',
#                         level=logging.INFO)

#     ## CUDA Init ##
#     args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     logging.info(f'Using {args.device}...')

#     ## DataLoader Config ##
#     args.dsa_param = ParamDiffAug()
#     args.dsa = True if args.method == 'DSA' else False
#     channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

#     if args.dsa:
#         logging.info('Augmentation : True')
#         logging.info('Augmentation Method : DSA')
#         logging.debug('DSA Param : \n'+args.dsa_param.print_param())


#     ## Training Config ##
#     args.outer_loop, args.inner_loop = get_loops(args.ipc)
#     eval_it_pool = np.arange(0, args.Iteration+1, args.iter_eval).tolist() if args.eval_mode == 'S' else [args.Iteration] # The list of iterations when we evaluate models and record results.
#     model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

#     #### Start of training ####
#     ''' organize the real dataset '''
#     images_all = []
#     labels_all = []
#     indices_class = [[] for c in range(num_classes)]

#     images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] # save images in (1,3,32,32) size
#     masks_all = torch.load('attention_mask.pt')
#     labels_all = [dst_train[i][1] for i in range(len(dst_train))]
#     for i, lab in enumerate(labels_all):    # Classify in classes
#         indices_class[lab].append(i)
#     ## All to CUDA
#     # images_all = torch.cat(images_all, dim=0).to(args.device)   # 50000x3x32x32
#     # masks_all = torch.cat(masks_all, dim=0).to(args.device)     # 50000x1x32x32
#     # labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device) # 50000x1
#     ## Batch to CUDA
#     images_all = torch.cat(images_all, dim=0)
#     masks_all = torch.cat(masks_all, dim=0)
#     labels_all = torch.tensor(labels_all, dtype=torch.long)

#     for c in range(num_classes):
#         logging.debug('class c = %d: %d real images'%(c, len(indices_class[c])))

#     def get_images(c, n, mix_rate=None): # get random n images(n = ipc) from class c
#         idx_shuffle = np.random.permutation(indices_class[c])[:n]
#         if mix_rate is None:
#             return images_all[idx_shuffle]
#         else:
#             # split idx_shuffle into given ratio
#             mix_idx = int(len(idx_shuffle) * mix_rate)
#             mixed = images_all[idx_shuffle[:mix_idx]] * masks_all[idx_shuffle[:mix_idx]]
#             unmixed = images_all[idx_shuffle[mix_idx:]]
#             return torch.cat([mixed, unmixed], dim=0)

#     for ch in range(channel):
#         logging.debug('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
        
#     accs_all_exps = dict() # record performances of all experiments
#     for key in model_eval_pool:
#         accs_all_exps[key] = []

#     for exp in range(args.num_exp):
#         logging.info('\n================== Exp %d ==================\n '%exp)
#         logging.info('Hyper-parameters: \n', args.__dict__)
#         logging.info('Evaluation model pool: ', model_eval_pool)
        
#         ''' initialize the synthetic data '''
#         image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
#         label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

#         if args.init == 'real':
#             logging.debug('initialize synthetic data from random real images')
#             for c in range(num_classes):
#                 image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().to(args.device).data
#         else:
#             logging.debug('initialize synthetic data from random noise')

#         ''' training '''
#         optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
#         optimizer_img.zero_grad()
#         criterion = nn.CrossEntropyLoss().to(args.device)
#         logging.info('%s training begins'%get_time())

#         for it in range(args.Iteration+1):

#             ''' Evaluate synthetic data '''
#             if it in eval_it_pool:
#                 for model_eval in model_eval_pool:
#                     logging.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
#                     if args.dsa:
#                         args.epoch_eval_train = 1000
#                         args.dc_aug_param = None
#                         logging.debug('DSA augmentation strategy: \n', args.dsa_strategy)
#                         logging.debug('DSA augmentation parameters: \n', args.dsa_param.__dict__)
#                     else:
#                         args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
#                         logging.debug('DC augmentation parameters: \n', args.dc_aug_param)

#                     if args.dsa or args.dc_aug_param['strategy'] != 'none':
#                         args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
#                     else:
#                         args.epoch_eval_train = 300

#                     accs = []
#                     for it_eval in range(args.num_eval):
#                         net_eval = get_network(args, model_eval, channel, num_classes, im_size).to(args.device) # get a random model
#                         image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
#                         net_eval, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
#                         accs.append(acc_test)
#                     logging.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

#                     if it == args.Iteration: # record the final results
#                         accs_all_exps[model_eval] += accs

#                 ''' visualize and save '''
#                 # if args.visualize:
#                 #     save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
#                 #     image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
#                 #     for ch in range(channel):
#                 #         image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
#                 #     image_syn_vis[image_syn_vis<0] = 0.0
#                 #     image_syn_vis[image_syn_vis>1] = 1.0
#                 #     save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

#             ''' Train synthetic data '''

#             # Gradient Matching
#             net = get_network(args, args.model, channel, num_classes, im_size).to(args.device) # get a random model
#             net.train()
#             net_parameters = list(net.parameters())
#             optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
#             optimizer_net.zero_grad()
#             loss_avg = 0
#             train_loss_avg = 0
#             train_acc_avg = 0
#             args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

#             for ol in range(args.outer_loop):

#                 ''' freeze the running mu and sigma for BatchNorm layers '''
#                 # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
#                 # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
#                 # This would make the training with BatchNorm layers easier.

#                 BN_flag = False
#                 BNSizePC = 32  # for batch normalization
#                 for module in net.modules():
#                     if 'BatchNorm' in module._get_name(): #BatchNorm
#                         BN_flag = True
#                 if BN_flag:
#                     img_real = torch.cat([get_images(c, BNSizePC, args.mask_prop) for c in range(num_classes)], dim=0).to(args.device)
#                     net.train() # for updating the mu, sigma of BatchNorm
#                     output_real = net(img_real) # get running mu, sigma
#                     for module in net.modules():
#                         if 'BatchNorm' in module._get_name():  #BatchNorm
#                             module.eval() # fix mu and sigma of every BatchNorm layer


#                 ''' update synthetic data '''
#                 losses = torch.tensor(0.0).to(args.device)
#                 for c in range(num_classes):
#                     img_real = get_images(c, args.batch_real, args.mask_prop)
#                     lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
#                     img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
#                     lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

#                     if args.dsa:
#                         seed = int(time.time() * 1000) % 100000
#                         img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
#                         img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

#                     output_real = net(img_real)
#                     loss_real = criterion(output_real, lab_real)
#                     gw_real = torch.autograd.grad(loss_real, net_parameters)
#                     gw_real = list((_.detach().clone() for _ in gw_real))

#                     output_syn = net(img_syn)
#                     loss_syn = criterion(output_syn, lab_syn)
#                     gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    
#                     losses += match_loss(gw_syn, gw_real, args)

#                 optimizer_img.zero_grad()
#                 losses.backward()
#                 optimizer_img.step()
#                 loss_avg += losses.item()
                
#                 ''' update network '''
#                 train_losses = 0
#                 train_accs = 0
#                 image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
#                 dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
#                 trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
#                 for il in range(args.inner_loop):
#                     train_loss, train_acc = epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)
#                     # train_losses += train_loss
#                     # train_accs += train_acc
#                 # train_loss_avg += train_losses / args.inner_loop
#                 # train_acc_avg += train_accs / args.inner_loop

#             # train_loss_avg /= (num_classes*args.outer_loop)
#             # train_acc_avg /= (num_classes*args.outer_loop)
#             loss_avg /= (num_classes*args.outer_loop)
            
#             # img_loss_vis.append(loss_avg)
#             # train_loss_vis.append(train_loss_avg)
#             # train_acc_vis.append(train_acc_avg)
            
#             logging.info('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            
#             if it == args.Iteration: # only record the final results
#                 data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
#                 torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc_exp%d.pt'%(args.method, args.dataset, args.model, args.ipc, exp)))
#         # visualize(img_loss_vis, train_loss_vis, train_acc_vis, args.save_path, 'vis_%s_%s_%s_%dipc_exp%d.png'%(args.method, args.dataset, args.model, args.ipc, exp))

#     logging.info('\n==================== Final Results ====================\n')
#     for key in model_eval_pool:
#         accs = accs_all_exps[key]
#         logging.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

# # def visualize(matching_loss, loss, acc, save_path, exp_name):
# #     import matplotlib.pyplot as plt
# #     fig, axs = plt.subplots(1,3,figsize=(15,4))
# #     outer_loop = list(range(1,len(matching_loss)+1))
# #     axs[0].plot(outer_loop, matching_loss, '-r')
# #     axs[0].set_title("Matching Loss")
# #     axs[1].plot(outer_loop, acc,'-g')
# #     axs[1].set_title("Accuracy")
# #     axs[2].plot(outer_loop, loss, '-r')
# #     axs[2].set_title("Training loss")
# #     plt.suptitle(exp_name.upper())
# #     plt.savefig(os.path.join(save_path, exp_name+'.png'))
# #     plt.close('all')

# if __name__ == '__main__':
#     parser = get_arguments()
#     config = parser.parse_args()
#     main(config)

# import os
# import logging
# import numpy as np
# import pytorch_lightning as pl
# import logging

# from data.data_module import DataModule

# def main(args):
#     # logger
#     run_name = "-".join([args.method, args.dataset, args.ipc, args.arch, args.comment])
#     wandb_logger = pl.loggers.WandbLogger(
#         save_dir=args.log_dir,
#         name=run_name,
#         project=args.project,
#         entity=args.entity,
#         offline=args.offline,
#     )
    
#     # build datamodule
#     augmentation = None
#     dm = DataModule(args.dataset, args.data_dir, args.batch_size, args.num_workers, augmentation)

#     # build model
#     model = Model(**args.__dict__)
    
#     # Train model
#     for exp in range(args.num_exp):
#         logging.info('\n================== Exp %d ==================\n '%exp)
#     trainer = pl.Trainer.from_argparse_args(
#         args, logger=wandb_logger, callbacks=[CheckpointCallback()], accelerator='dp'
#     )
#     trainer.fit(model, dm)


# if __name__ == "__main__":
#     parser = pl.Trainer.add_argparse_args(parser)
#     args = parser.parse_args()

#     main(args)