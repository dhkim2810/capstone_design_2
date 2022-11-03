import os
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import models, datasets

data_dir = "./datasets"
result_dir = "./results"

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
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

    def forward(self, x, at=False):
        out = self.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if at:
            return out, (out1, out2, out3, out4)
        return out

def resnet18(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, 4),
                        tv.transforms.RandomHorizontalFlip(0.3),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean=mean, std=std)
                ])
    dst_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform) # no augmentation
    dst_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=4)
    
    # Model
    net = resnet18(3,10)
    source = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
    del source['conv1.weight'], source['fc.weight'], source['fc.bias']
    net.load_state_dict(source, strict=False)
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
    torch.save(net.state_dict(), os.path.join(result_dir, "resnet18_cifar.pt"))

if __name__=="__main__":
    main()