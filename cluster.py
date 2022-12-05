import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# K-Means
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained

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

def cluster(data, num_cluster, layer_idx, norm, cluster_path, iteration=1000, viz=True):
    
    # Directory Setup
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)
    
    # data
    images_all, labels_all, indices_class, class_names = data

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18(3, 10)
    state_dict_path = os.path.join(cluster_path, "resnet18_cifar.pt")
    if not os.path.exists(state_dict_path):
        tmp_resnet18 = resnet18(pretrained=True)
        state_dict = tmp_resnet18.state_dict()
        del state_dict['conv1.weight'], state_dict['fc.weight'], state_dict['fc.bias']
        net.load_state_dict(state_dict)
        torch.save(net.state_dict(), state_dict_path)
    else:
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    
    # Clustering
    indices_sub_class = {}
    sub_class_index = {i:{} for i in range(10)}
    indices_sub_class_centers = {}
    
    for class_idx in range(10):
        class_name = class_names[class_idx]
        batch_size = int(len(indices_class[class_idx]) / iteration)
        cluster_size = int(len(indices_class[class_idx]) / num_cluster)
        
        features = []
        print(f"Gathering features for class {class_name}")
        for it in range(iteration):
            imgs = images_all[indices_class[class_idx][it*batch_size:(it+1)*batch_size]]
            _, _feature = net(imgs)
            features.append(F.avg_pool2d(_feature[layer_idx], (_feature[layer_idx].shape[2],_feature[layer_idx].shape[3])).squeeze())
        features = torch.cat(features, dim=0).detach().cpu().numpy()
        
        print(f"Calculating cluster center features for class {class_name}")
        if norm:
            
            clf = KMeansConstrained(n_clusters=num_cluster, size_min=cluster_size-50, size_max=cluster_size+50, random_state=1)
            clf.fit(features)
            labels = clf.labels_
            centers = clf.cluster_centers_
        else:
            kmeans = KMeans(n_clusters=num_cluster, n_init=5, max_iter=300, random_state=1, verbose=0).fit(features)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
        
        indices_sub_class[class_idx] = labels
        indices_sub_class_centers[class_idx] = centers
        
        feature_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=30).fit_transform(np.concatenate((features, centers),axis=0))
        feature = feature_embedded[:5000, :]
        centers = feature_embedded[5000:,:]
        
        df = pd.DataFrame(dict(x1=feature[:,0], x2=feature[:,1], label=labels))
        cdict = {i:plt.cm.tab20(i) for i in range(num_cluster)}

        if viz:
            fig, ax = plt.subplots(figsize=(10,6))
            grouped = df.groupby('label')
            for key, group in grouped:
                group.plot(ax=ax, kind='scatter', x='x1',y='x2',color=cdict[key])
            fig.suptitle(f"CIFAR10 \"{class_name} Class\" Kmeans Clustering - k={num_cluster}")
            fig_name = os.path.join(cluster_path, f"Viz/cifar10_k{num_cluster}_{layer_idx}_{class_name}")
            if norm:
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
    torch.save(sub_class_index, os.path.join(cluster_path, f"class_idx_cifar10_k{num_cluster}_{layer_idx}_{norm}.pt"))
    return sub_class_index