import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from data import get_dataset

def main(args):
    # set seed
    torch.manual_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set dataset
    _, _, num_classes, class_names, _, _, _, _, testloader = get_dataset(args.dataset, args.data_dir)

    # set model
    # net = models.resnet18(pretrained=True)
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net = net.to(device)
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    
    # generate features
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(testloader)):
                print(idx+1, '/', len(testloader))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)
    
    tsne_plot(targets, outputs, args.save_dir, args.save_name, num_classes, class_names)

def tsne_plot(targets, outputs, save_dir, save_name, num_classes, class_names):
    print('generating t-SNE plot...')
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        alpha=0.5
    )
    plt.legend(labels=class_names, bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,f'{save_name}.png'), bbox_inches='tight')
    print('done!')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch t-SNE for CIFAR10')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to run tSNE evaluation')
    parser.add_argument('--data_dir', type=str, default='./data/CIFAR', help='path to dataset')
    parser.add_argument('--save-dir', type=str, default='./results/t_sne', help='path to save the t-sne image')
    parser.add_argument('--save_name', type=str, default='t_sne_cifar10', help='name of result file')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')
    args = parser.parse_args()
    main(args)