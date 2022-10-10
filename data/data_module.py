import numpy as np

import torch
from torchvision import datasets, transforms


class DataModule():
    def __init__(self, data_dir, dataset, **kwargs):
        self.data_dir = data_dir
        self.dataset = getattr(datasets, dataset)
        self.batch_size = kwargs.get('batch_size', 256)
        self.num_workers = kwargs.get('num_workers', 4)
        self.prepare_data(dataset)


    def prepare_data(self, dataset):
        if dataset == 'MNIST':
            channel = 1
            im_size = (28, 28)
            num_classes = 10
            mean = [0.1307]
            std = [0.3081]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            self.dst_train = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform) # no augmentation
            self.dst_test = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
            class_names = [str(c) for c in range(num_classes)]
            
        elif dataset == 'FashionMNIST':
            channel = 1
            im_size = (28, 28)
            num_classes = 10
            mean = [0.2861]
            std = [0.3530]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            self.dst_train = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=transform) # no augmentation
            self.dst_test = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=transform)
            class_names = self.dst_train.classes

        elif dataset == 'SVHN':
            channel = 3
            im_size = (32, 32)
            num_classes = 10
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            self.dst_train = datasets.SVHN(self.data_dir, split='train', download=True, transform=transform) # no augmentation
            self.dst_test = datasets.SVHN(self.data_dir, split='test', download=True, transform=transform)
            class_names = [str(c) for c in range(num_classes)]

        elif dataset == 'CIFAR10':
            channel = 3
            im_size = (32, 32)
            num_classes = 10
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            self.dst_train = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform) # no augmentation
            self.dst_test = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform)
            class_names = self.dst_train.classes

        elif self.dataset == 'CIFAR100':
            channel = 3
            im_size = (32, 32)
            num_classes = 100
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            self.dst_train = datasets.CIFAR100(self.data_dir, train=True, download=True, transform=transform) # no augmentation
            self.dst_test = datasets.CIFAR100(self.data_dir, train=False, download=True, transform=transform)
            class_names = self.dst_train.classes
            
        else:
            exit('unknown dataset: %s'%self.dataset)
        
        self.labeled_data = {i:{} for i in range(num_classes)}
        for i in range(len(self.dst_train)):
            self.labeled_data[self.dst_train[i][1]].append(torch.unsqueeze(self.dst_train[i][0], dim=0))
        for class_idx, items in self.labeled_data.items():
            self.labeled_data[class_idx] = torch.cat(items, dim=0)
        self.dataset_config = {'channel':channel,'im_size':im_size,'num_classes':num_classes,'class_names':class_names,'mean':mean,'std':std}


    def get_dataset_config(self):
        return self.dataset_config


    def get_real_images(self, class_idx, n):
        idx_shuffle = np.random.permutation(self.labeled_data[class_idx].size(0))[:n]
        return self.labeled_data[class_idx][idx_shuffle]