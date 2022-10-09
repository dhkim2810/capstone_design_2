import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, **kwargs):
        super().__init__()
        self.data_dir = kwargs.get('data_dir', './data')
        self.batch_size = kwargs.get('batch_size', 256)
        self.num_workers = kwargs.get('num_workers', 4)
        self.augmentation = kwargs.get('augmentation', None)
        self.dataset = getattr(datasets, dataset)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        self.dataset(self.dataset_config['data_dir'], train=True, download=True)
        self.dataset(self.dataset_config['data_dir'], train=False, download=True)
        self.train_transform = None
        self.val_transform = None

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset_train = self.dataset(self.data_dir, train=True, transform=self.train_transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "val":
            self.dataset_val = self.dataset(self.data_dir, train=False, transform=self._transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch)





class DataModule():
    def __init__(self, dataset, transform, config):
        self.dataset = dataset
        self.transform = None
        self.transform_config = transform
        self.configs = config
    
    def get_loaders(self):
        dst_train, dst_test, data_metadata = self.get_dataset()
        
        return dst_train, dst_test

    def get_dataset(self):
        self.set_augmentation()
        if self.dataset == 'MNIST':
            channel = 1
            im_size = (28, 28)
            num_classes = 10
            mean = [0.1307]
            std = [0.3081]
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if self.transform is None else self.transform
            dst_train = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform) # no augmentation
            dst_test = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform)
            class_names = [str(c) for c in range(num_classes)]
            
        elif self.dataset == 'FashionMNIST':
            channel = 1
            im_size = (28, 28)
            num_classes = 10
            mean = [0.2861]
            std = [0.3530]
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if self.transform is None else self.transform
            dst_train = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=self.transform) # no augmentation
            dst_test = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=self.transform)
            class_names = dst_train.classes

        elif self.dataset == 'SVHN':
            channel = 3
            im_size = (32, 32)
            num_classes = 10
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if self.transform is None else self.transform
            dst_train = datasets.SVHN(self.data_dir, train=True, download=True, transform=self.transform) # no augmentation
            dst_test = datasets.SVHN(self.data_dir, train=False, download=True, transform=self.transform)
            class_names = [str(c) for c in range(num_classes)]

        elif self.dataset == 'CIFAR10':
            channel = 3
            im_size = (32, 32)
            num_classes = 10
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if self.transform is None else self.transform
            dst_train = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=self.transform) # no augmentation
            dst_test = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=self.transform)
            class_names = dst_train.classes

        elif self.dataset == 'CIFAR100':
            channel = 3
            im_size = (32, 32)
            num_classes = 100
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if self.transform is None else self.transform
            dst_train = datasets.CIFAR100(self.data_dir, train=True, download=True, transform=self.transform) # no augmentation
            dst_test = datasets.CIFAR100(self.data_dir, train=False, download=True, transform=self.transform)
            class_names = dst_train.classes

        else:
            exit('unknown dataset: %s'%self.dataset)
        
        return dst_train, dst_test, {'channel':channel,'im_size':im_size,'num_classes':num_classes,'mean':mean,'std':std,'class_names':class_names}




def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x