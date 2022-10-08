import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu(self.fc_1(out))
        out = self.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out