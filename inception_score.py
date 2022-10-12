import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig, target_class):
        self.imgs = [orig[i][0] for i in range(len(orig)) if orig[i][1] == target_class]

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':

    cifar = dataset.CIFAR10(root='./datasets', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                             ])
    )
    
    incep = []
    std = []
    for target_class in range(10):
        dset = IgnoreLabelDataset(cifar, target_class)
        print("Target Class : %d \(%d imgs\)", target_class, )
        mean, _std = inception_score(dset, cuda=False, batch_size=32, resize=True, splits=10)
        incep.append(mean)
        std.append(_std)
    
    print("Overall Result for CIFAR10")
    for idx, (a,b) in enumerate(list(zip(incep, std))):
        print(f"Class {idx} : {a:.2f} +- {b:.2f}")

'''
Overall Result for CIFAR10
Class 0 : 4.90 +- 0.21
Class 1 : 3.19 +- 0.11
Class 2 : 5.81 +- 0.16
Class 3 : 6.44 +- 0.21
Class 4 : 4.70 +- 0.22
Class 5 : 7.78 +- 0.29
Class 6 : 5.06 +- 0.11
Class 7 : 3.98 +- 0.18
Class 8 : 4.44 +- 0.19
Class 9 : 2.70 +- 0.08
'''