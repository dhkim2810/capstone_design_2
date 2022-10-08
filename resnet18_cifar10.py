import torch
import torchvision as tv
from torchvision import models, datasets

import argparse

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform) # no augmentation
    dst_test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4)
    
    # Model
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if torch.cuda.is_available():
        net = net.to(device)
    
    # Configuration
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # Train
    num_epochs = 200
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        epoch_loss /= idx
        
        net.eval()
        total = correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net(inputs)
            correct += (outputs == labels).sum().item()
            total += labels.size(0)
        acc = correct/total
        if epoch % 5 == 0:
            print(f"[{epoch}/{num_epochs}] Train Loss : {epoch_loss:.4}\tTest Acc : {acc:2.2f}")
    
    # Save
    torch.save(net.state_dict(), args.save_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to run classification')
    parser.add_argument('--data_dir', type=str, default='./data/CIFAR', help='path to dataset')
    parser.add_argument('--save-dir', type=str, default='./results', help='path to save the results')
    args = parser.parse_args()
    main(args)