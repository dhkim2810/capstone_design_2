import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

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

# Directory Setup
data_dir = "./dataset"
result_dir = "./results"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
''' organize the real dataset '''
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
dst_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform) # no augmentation
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
num_layer = 4
net = ResNet18(3, 10)
state_dict = torch.load("./results/resnet18_cifar.pt", map_location='cpu')
net.load_state_dict(state_dict)
net = net.to(device)
net.eval()

# K-Means
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

def get_feature(layer_idx):
    class_feature = []
    for class_idx in range(10):
        class_name = class_names[class_idx]
        batch_size = int(len(indices_class[class_idx]) / 20)
        
        features = []
        print(f"Gathering features for class {class_name}")
        for it in range(20):
            imgs = images_all[indices_class[class_idx][it*batch_size:(it+1)*batch_size]]
            _, _feature = net(imgs)
            features.append(F.avg_pool2d(_feature[layer_idx], (_feature[layer_idx].shape[2],_feature[layer_idx].shape[3])).squeeze())
        features = torch.cat(features, dim=0).detach().cpu().numpy()
        class_feature.append(features)
    return class_feature

def cluster(num_clusters, features, layer_idx):
    indices_sub_class = {}
    sub_class_index = {i:{} for i in range(10)}
    indices_sub_class_centers = {}
    for class_idx in range(10):
        class_name = class_names[class_idx]
        print(f"Calculating cluster center features for class {class_name}")
        size = int(5000 / num_clusters)
        clf = KMeansConstrained(n_clusters=num_clusters, size_min=size-100, size_max=size+100, random_state=1)
        clf.fit(features[class_idx])
        labels = clf.labels_
        centers = clf.cluster_centers_
        # kmeans = KMeans(n_clusters=num_clusters, n_init=5, max_iter=300, random_state=1, verbose=0).fit(features)
        # labels = kmeans.labels_
        # centers = kmeans.cluster_centers_
        
        indices_sub_class[class_idx] = labels
        indices_sub_class_centers[class_idx] = centers
        
        feature_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=30).fit_transform(np.concatenate((features[class_idx], centers),axis=0))
        feature = feature_embedded[:5000, :]
        centers = feature_embedded[5000:,:]
        
        df = pd.DataFrame(dict(x1=feature[:,0], x2=feature[:,1], label=labels))

        cdict = {i:plt.cm.tab20(i) for i in range(num_clusters)}

        fig, ax = plt.subplots(figsize=(10,6))
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x1',y='x2',color=cdict[key])
        fig.suptitle(f"CIFAR10 \"{class_name} Class\" Kmeans Clustering - k={num_clusters}")
        plt.savefig(f"./clustering/cifar10_k{num_clusters}_{layer_idx}_{class_name}_norm")
        
        original_index = indices_class[class_idx]
        sub_index = indices_sub_class[class_idx]
        for orig, sub in zip(original_index, sub_index):
            if sub not in sub_class_index[class_idx]:
                sub_class_index[class_idx][sub] = [orig]
            else:
                sub_class_index[class_idx][sub].append(orig)
    torch.save(sub_class_index, f"./clustering/class_idx_cifar10_k{num_clusters}_{layer_idx}_norm.pt")


for layer in [2,3,1]:
    feature = get_feature(layer)
    for k in [5,10,15,20,25]:
        cluster(k, feature, layer)