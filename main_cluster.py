import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# K-Means
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, [out1, out2, out3, out4]

def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--iteration', type=int, default=20, help='training iterations')
    parser.add_argument('--data_path', type=str, default='dataset', help='dataset path')
    parser.add_argument('--cluster_path', type=str, default='clustering', help='path to save results')
    parser.add_argument('--layer_idx', type=int, default=1, help='layer of subclass')
    parser.add_argument('--num_cluster', type=int, default=20)
    parser.add_argument('--norm', action='store_true', default=False)
    parser.add_argument('--viz', action='store_true', default=False)
    args = parser.parse_args()
    
    # Directory Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    ''' organize the real dataset '''
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform) # no augmentation
    class_names = dst_train.classes

    images_all = []
    labels_all = []
    indices_class = [[] for c in range(10)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    # Model
    net = ResNet18(3, 10)
    state_dict = torch.load(os.path.join(args.cluster_path, "resnet18_cifar.pt"), map_location='cpu')
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    
    # Clustering
    indices_sub_class = {}
    sub_class_index = {i:{} for i in range(10)}
    indices_sub_class_centers = {}
    
    for class_idx in range(10):
        class_name = class_names[class_idx]
        batch_size = int(len(indices_class[class_idx]) / args.iteration)
        cluster_size = int(len(indices_class[class_idx]) / args.num_cluster)
        
        features = []
        print(f"Gathering features for class {class_name}")
        for it in range(args.iteration):
            imgs = images_all[indices_class[class_idx][it*batch_size:(it+1)*batch_size]]
            _, _feature = net(imgs)
            features.append(F.avg_pool2d(_feature[args.layer_idx], (_feature[args.layer_idx].shape[2],_feature[args.layer_idx].shape[3])).squeeze())
        features = torch.cat(features, dim=0).detach().cpu().numpy()
        
        print(f"Calculating cluster center features for class {class_name}")
        if args.norm:
            
            clf = KMeansConstrained(n_clusters=args.num_cluster, size_min=cluster_size-50, size_max=cluster_size+50, random_state=1)
            clf.fit(features)
            labels = clf.labels_
            centers = clf.cluster_centers_
        else:
            kmeans = KMeans(n_clusters=args.num_cluster, n_init=5, max_iter=300, random_state=1, verbose=0).fit(features)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
        
        indices_sub_class[class_idx] = labels
        indices_sub_class_centers[class_idx] = centers
        
        feature_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=30).fit_transform(np.concatenate((features, centers),axis=0))
        feature = feature_embedded[:5000, :]
        centers = feature_embedded[5000:,:]
        
        df = pd.DataFrame(dict(x1=feature[:,0], x2=feature[:,1], label=labels))
        cdict = {i:plt.cm.tab20(i) for i in range(args.num_cluster)}

        if args.viz:
            fig, ax = plt.subplots(figsize=(10,6))
            grouped = df.groupby('label')
            for key, group in grouped:
                group.plot(ax=ax, kind='scatter', x='x1',y='x2',color=cdict[key])
            fig.suptitle(f"CIFAR10 \"{class_name} Class\" Kmeans Clustering - k={args.num_cluster}")
            fig_name = os.path.join(args.cluster_path, f"Viz/cifar10_k{args.num_cluster}_{args.layer_idx}_{class_name}")
            if args.norm:
                fig_name += '_norm'
            plt.savefig(fig_name)
            plt.clf()
        
        # Organize labels for Dataset Condensation
        original_index = indices_class[class_idx]
        sub_index = indices_sub_class[class_idx]
        for orig, sub in zip(original_index, sub_index):
            if sub not in sub_class_index[class_idx]:
                sub_class_index[class_idx][sub] = [orig]
            else:
                sub_class_index[class_idx][sub].append(orig)
    torch.save(sub_class_index, os.path.join(args.cluster_path, f"class_idx_cifar10_k{args.num_cluster}_{args.layer_idx}_{args.norm}.pt"))

if __name__=="__main__":
    main()