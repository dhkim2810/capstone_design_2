import os
from random import shuffle
from this import d
from venv import create
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .mlp import MLP
from .lenet import LeNet
from .alexnet import AlexNet
from .convnet import ConvNet
from .vgg import VGG
from .resnet import ResNet18, ResNet18BN, ResNet18_AP, ResNet18BN_AP, ResNet34, ResNet50, ResNet101, ResNet152

class ModelModule():
    def __init__(self, model, aug_eval=None, channel=3, num_classes=10, img_size=(32,32), **kwargs):
        self.__dict__ = {k: v for (k, v) in kwargs.items() if not callable(v)}
        self.model = model
        self.augmentation = aug_eval
        self.channel = channel
        self.num_classes = num_classes
        self.img_size = img_size
        self.set_model()
        self.net.train()


    def set_model(self):
        if self.model == 'MLP':
            self.net = MLP(self.channel, self.num_classes)
        elif self.model == 'LeNet':
            self.net = LeNet(self.channel, self.num_classes)
        elif self.model == 'AlexNet':
            self.net = AlexNet(self.channel, self.num_classes)
        elif 'ConvNet' in self.model:
            net_width, net_depth, net_act, net_norm, net_pooling = self.get_ConvNet_config()
            self.net = ConvNet(self.channel, self.num_classes, net_width, net_depth, net_act, net_norm, net_pooling, self.img_size)
        elif 'VGG' in self.model:
            self.net = VGG(self.model, self.channel, self.num_classes, norm='batchnorm' if 'BN' in self.model else 'instancenorm')
        elif self.model == 'ResNet18':
            self.net = ResNet18(self.channel, self.num_classes)
        elif self.model == 'ResNet18BN':
            self.net = ResNet18BN(self.channel, self.num_classes)
        elif self.model == 'ResNet18BN_AP':
            self.net = ResNet18BN_AP(self.channel, self.num_classes)
        
        self.net.to(self.device)
        self.net_parameters = list(self.net.parameters())
        self.optimizer_net = torch.optim.SGD(self.net.parameters(), lr=self.lr_net, momentum=self.opt_momentum, weight_decay=self.opt_wd)
        self.optimizer_net.zero_grad()


    def freeze_model_BN_layers(self, inputs, BNSizePC):
        BN_flag = False
        for module in self.net.modules():
            if 'BatchNorm' in module._get_name():
                BN_flag = True
                break
        if BN_flag:
            _ = self.net(inputs)
            for module in self.net.modules():
                if 'BatchNorm' in module._get_name():
                    module.eval()
        return BN_flag


    def calculate_matching_loss(self, img_real, lab_real, img_syn, lab_syn):
        output_real = self.net(img_real)
        loss_real = F.cross_entropy(output_real, lab_real)
        gw_real = torch.autograd.grad(loss_real, self.net_parameters)
        gw_real = list((_.detach().clone() for _ in gw_real))
        
        output_syn = self.net(img_syn)
        loss_syn = F.cross_entropy(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, self.net_parameters, create_graph=True)
        
        return self.matching_gradient_loss(gw_real, gw_syn)


    def epoch(self, loader, aug=False, mode='train'):        
        losses = 0.0
        accs = 0.0
        for idx, (img, lab) in enumerate(loader):
            img = img.to(self.device)
            lab = lab.to(self.device)
            
            if aug:
                img = self.augmentation.augment(img)
            
            output = self.net(img)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
            accs += acc
            
            if mode == 'train':
                loss = self.cross_entropy_loss(output, lab)
                losses += loss.item() * img.size(0)
                
                self.optimizer_net.zero_grad()
                loss.backward()
                self.optimizer_net.step()
        losses /= img.size(0)
        accs /= img.size(0)
        return losses, accs


    def train_with_synthetic_data(self, imgs, labs, epochs=1000, batch_size=32, lr_schedule=False, aug=False, freeze=False):
        if not freeze:
            self.net.train()
        dst = TensorDataset(imgs, labs)
        loader = DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=0)
        
        lr = float(self.lr_net)
        lr_scheduler = [epochs//2+1] if lr_schedule else None
        
        loss_avg = []
        acc_avg = []
        for epoch in range(epochs):
            loss, acc = self.epoch(loader, aug, mode='train')
            loss_avg.append(loss)
            acc_avg.append(acc)
            if epoch in lr_scheduler and lr_schedule:
                lr *= 0.1
                for g in self.optimizer_net.param_groups:
                    g['lr'] = lr

        return loss_avg, acc_avg


    def test_with_synthetic_data(self, imgs, labs, batch_size=32):
        self.net.eval()
        dst = TensorDataset(imgs, labs)
        loader = DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=0)
        loss, acc = self.epoch(loader, mode='test')
        return loss, acc


    def cross_entropy_loss(self, inputs, targets):
        return F.cross_entropy(inputs, targets)


    def distance_wb(self, gwr, gws):
        shape = gwr.shape
        if len(shape) == 4: # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2: # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return torch.tensor(0, dtype=torch.float).to(gwr)

        dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis


    def matching_gradient_loss(self, gw_real, gw_syn):
        dis = torch.tensor(0.0).to(self.device)

        if self.dis_metric == 'wb':
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += self.distance_wb(gwr, gws)

        elif self.dis_metric == 'mse':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

        elif self.dis_metric == 'cos':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        else:
            exit('unknown distance function: %s'%self.dis_metric)

        return dis


    def attention(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_loss(self, x, y):
        # MSE 
        return (self.attention(x) - self.attention(y)).pow(2).mean()


    def get_ConvNet_config(self):
        net_width = 128
        net_depth = 3
        net_act = 'relu'
        net_norm = 'instancenorm'
        net_pooling = 'avgpooling'
        
        if self.model ==  'ConvNetD1':
            net_depth = 1
        elif self.model == 'ConvNetD2':
            net_depth = 2
        elif self.model == 'ConvNetD3':
            net_depth = 3
        elif self.model == 'ConvNetD4':
            net_depth = 4
            
        elif self.model == 'ConvNetW32':
            net_width = 32
        elif self.model == 'ConvNetW64':
            net_width = 64
        elif self.model == 'ConvNetW128':
            net_width = 128
        elif self.model == 'ConvNetW256':
            net_width = 256

        elif self.model == 'ConvNetAS':
            net_act = 'sigmoid'
        elif self.model == 'ConvNetAR':
            net_act = 'relu'
        elif self.model == 'ConvNetAL':
            net_act = 'leakyrelu'

        elif self.model == 'ConvNetNN':
            net_norm = 'none'
        elif self.model == 'ConvNetBN':
            net_norm='batchnorm'
        elif self.model == 'ConvNetLN':
            net_norm='layernorm'
        elif self.model == 'ConvNetIN':
            net_norm='instancenorm'
        elif self.model == 'ConvNetGN':
            net_norm='groupnorm'

        elif self.model == 'ConvNetNP':
            net_pooling='none'
        elif self.model == 'ConvNetMP':
            net_pooling='maxpooling'
        elif self.model == 'ConvNetAP':
            net_pooling='avgpooling'
        
        return net_width, net_depth, net_act, net_norm, net_pooling


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]