import sys
sys.path.append(".")
import torch
import torch.nn as nn
from torch.distributions import Normal


def loss_function(x, x_hat, mean, var):
    
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ torch.log(var.pow(2)) - mean.pow(2) - var.pow(2))

    return reproduction_loss + KLD