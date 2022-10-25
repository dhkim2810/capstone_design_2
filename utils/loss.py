import torch
import torch.nn.functional as F

def LFA(output, target):
    '''
    Layer-wise Feature Alignment with Discriminative Loss
    '''
    lfa_loss = 0.0
    for syn, real in zip(output, target):
        _syn = syn.mean(dim=0)
        _real = real.mean(dim=0)
        lfa_loss += F.mse_loss(_syn, _real)
    
    return lfa_loss


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def AT_Loss(output, target):
    return (at(output) - at(target)).pow(2).mean()


def LAA(output, target):
    '''
    Layer-wise Attention Alignment
    '''
    loss = 0.0
    for syn, real in zip(output, target):
        loss += AT_Loss(syn, real)
    return loss


def DL(real, syn_features):
    '''
    Discrimination Loss
    '''
    K,N = real.shape[:2]
    real = real.view(K,N,-1)
    syn_features = syn_features.view(K,-1)
    O = torch.inner(real, syn_features).view(-1, K)
    loss = -torch.mean(torch.log(F.softmax(O)))
    return loss