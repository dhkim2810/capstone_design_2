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
        self.class_name = orig.classes
        self.imgs = [orig[i][0] for i in range(len(orig)) if orig[i][1] == target_class]

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)
    
    def get_class_name(self, target_class):
        return self.class_name[target_class]

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

    cifar = dataset.CIFAR100(root='./datasets', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                             ])
    )
    
    incep = []
    std = []
    for target_class in range(100):
        class_name = cifar.classes[target_class]
        dset = IgnoreLabelDataset(cifar, target_class)
        mean, _std = inception_score(dset, cuda=False, batch_size=64, resize=True, splits=10)
        print(f"[{target_class}] Class {class_name}\t: {mean:.2f} +- {_std:.2f} ({len(dset)} imgs)")
        incep.append(mean)
        std.append(_std)
    
    print("Overall Result for CIFAR10")
    for idx, (a,b) in enumerate(list(zip(incep, std))):
        print(f"Class {idx}\t: {a:.2f} +- {b:.2f}")

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

Overall Result for CIFAR100
[0] Class apple         : 3.62 +- 0.43 (500 imgs)
[1] Class aquarium_fish : 3.63 +- 0.41 (500 imgs)
[2] Class baby          : 4.40 +- 0.60 (500 imgs)
[3] Class bear          : 5.11 +- 0.59 (500 imgs)
[4] Class beaver        : 4.38 +- 0.33 (500 imgs)
[5] Class bed           : 3.45 +- 0.48 (500 imgs)
[6] Class bee           : 5.03 +- 0.52 (500 imgs)
[7] Class beetle        : 4.50 +- 0.41 (500 imgs)
[8] Class bicycle       : 3.80 +- 0.23 (500 imgs)
[9] Class bottle        : 3.17 +- 0.30 (500 imgs)
[10] Class bowl         : 4.84 +- 0.36 (500 imgs)
[11] Class boy          : 4.58 +- 0.54 (500 imgs)
[12] Class bridge       : 4.03 +- 0.18 (500 imgs)
[13] Class bus          : 3.01 +- 0.24 (500 imgs)
[14] Class butterfly    : 4.65 +- 0.41 (500 imgs)
[15] Class camel        : 4.12 +- 0.37 (500 imgs)
[16] Class can          : 3.76 +- 0.39 (500 imgs)
[17] Class castle       : 3.27 +- 0.29 (500 imgs)
[18] Class caterpillar  : 5.02 +- 0.36 (500 imgs)
[19] Class cattle       : 4.49 +- 0.39 (500 imgs)
[20] Class chair        : 2.92 +- 0.29 (500 imgs)
[21] Class chimpanzee   : 3.44 +- 0.39 (500 imgs)
[22] Class clock        : 3.65 +- 0.47 (500 imgs)
[23] Class cloud        : 3.21 +- 0.38 (500 imgs)
[24] Class cockroach    : 3.21 +- 0.25 (500 imgs)
[25] Class couch        : 4.26 +- 0.68 (500 imgs)
[26] Class crab         : 4.90 +- 0.53 (500 imgs)
[27] Class crocodile    : 4.27 +- 0.37 (500 imgs)
[28] Class cup          : 3.40 +- 0.32 (500 imgs)
[29] Class dinosaur     : 4.62 +- 0.55 (500 imgs)
[30] Class dolphin      : 3.15 +- 0.19 (500 imgs)
[31] Class elephant     : 2.62 +- 0.23 (500 imgs)
[32] Class flatfish     : 5.17 +- 0.39 (500 imgs)
[33] Class forest       : 3.52 +- 0.38 (500 imgs)
[34] Class fox          : 3.17 +- 0.30 (500 imgs)
[35] Class girl         : 4.11 +- 0.45 (500 imgs)
[36] Class hamster      : 4.05 +- 0.44 (500 imgs)
[37] Class house        : 3.63 +- 0.34 (500 imgs)
[38] Class kangaroo     : 4.40 +- 0.39 (500 imgs)
[39] Class keyboard     : 3.78 +- 0.50 (500 imgs)
[40] Class lamp         : 4.12 +- 0.61 (500 imgs)
[41] Class lawn_mower   : 3.50 +- 0.29 (500 imgs)
[42] Class leopard      : 4.79 +- 0.48 (500 imgs)
[43] Class lion         : 3.37 +- 0.27 (500 imgs)
[44] Class lizard       : 5.22 +- 0.34 (500 imgs)
[45] Class lobster      : 5.23 +- 0.20 (500 imgs)
[46] Class man          : 3.90 +- 0.37 (500 imgs)
[47] Class maple_tree   : 2.84 +- 0.16 (500 imgs)
[48] Class motorcycle   : 3.29 +- 0.39 (500 imgs)
[49] Class mountain     : 2.74 +- 0.17 (500 imgs)
[50] Class mouse        : 4.58 +- 0.33 (500 imgs)
[51] Class mushroom     : 4.13 +- 0.33 (500 imgs)
[52] Class oak_tree     : 2.10 +- 0.08 (500 imgs)
[53] Class orange       : 3.19 +- 0.21 (500 imgs)
[54] Class orchid       : 3.95 +- 0.45 (500 imgs)
[55] Class otter        : 5.03 +- 0.48 (500 imgs)
[56] Class palm_tree    : 3.24 +- 0.35 (500 imgs)
[57] Class pear         : 4.42 +- 0.52 (500 imgs)
[58] Class pickup_truck : 2.78 +- 0.27 (500 imgs)
[59] Class pine_tree    : 2.96 +- 0.23 (500 imgs)
[60] Class plain        : 2.91 +- 0.23 (500 imgs)
[61] Class plate        : 3.88 +- 0.46 (500 imgs)
[62] Class poppy        : 3.71 +- 0.41 (500 imgs)
[63] Class porcupine    : 4.36 +- 0.45 (500 imgs)
[64] Class possum       : 4.41 +- 0.28 (500 imgs)
[65] Class rabbit       : 5.16 +- 0.46 (500 imgs)
[66] Class raccoon      : 4.39 +- 0.35 (500 imgs)
[67] Class ray          : 4.22 +- 0.26 (500 imgs)
[68] Class road         : 3.29 +- 0.33 (500 imgs)
[69] Class rocket       : 3.89 +- 0.47 (500 imgs)
[70] Class rose         : 4.13 +- 0.48 (500 imgs)
[71] Class sea          : 3.34 +- 0.32 (500 imgs)
[72] Class seal         : 5.14 +- 0.58 (500 imgs)
[73] Class shark        : 3.27 +- 0.28 (500 imgs)
[74] Class shrew        : 4.39 +- 0.51 (500 imgs)
[75] Class skunk        : 3.17 +- 0.37 (500 imgs)
[76] Class skyscraper   : 4.28 +- 0.25 (500 imgs)
[77] Class snail        : 4.71 +- 0.32 (500 imgs)
[78] Class snake        : 4.31 +- 0.40 (500 imgs)
[79] Class spider       : 4.20 +- 0.34 (500 imgs)
[80] Class squirrel     : 3.75 +- 0.29 (500 imgs)
[81] Class streetcar    : 3.50 +- 0.34 (500 imgs)
[82] Class sunflower    : 3.55 +- 0.30 (500 imgs)
[83] Class sweet_pepper : 4.35 +- 0.44 (500 imgs)
[84] Class table        : 4.89 +- 0.34 (500 imgs)
[85] Class tank         : 3.41 +- 0.37 (500 imgs)
[86] Class telephone    : 4.02 +- 0.50 (500 imgs)
[87] Class television   : 3.29 +- 0.30 (500 imgs)
[88] Class tiger        : 3.67 +- 0.33 (500 imgs)
[89] Class tractor      : 2.82 +- 0.28 (500 imgs)
[90] Class train        : 4.12 +- 0.40 (500 imgs)
[91] Class trout        : 4.56 +- 0.43 (500 imgs)
[92] Class tulip        : 4.38 +- 0.34 (500 imgs)
[93] Class turtle       : 4.50 +- 0.29 (500 imgs)
[94] Class wardrobe     : 3.23 +- 0.28 (500 imgs)
[95] Class whale        : 3.16 +- 0.25 (500 imgs)
[96] Class willow_tree  : 2.94 +- 0.25 (500 imgs)
[97] Class wolf         : 3.76 +- 0.46 (500 imgs)
[98] Class woman        : 4.17 +- 0.49 (500 imgs)
[99] Class worm         : 3.29 +- 0.40 (500 imgs)
'''