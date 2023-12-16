import sys
sys.path.append(".")
import torch
import torch.nn as nn
from torch.distributions import Normal


def loss_function(rna, rna_generated, atac, atac_generated, mean, var):
    
    reproduction_loss_rna = nn.functional.binary_cross_entropy(rna_generated, rna, reduction='sum')
    reproduction_loss_atac = nn.functional.binary_cross_entropy(atac_generated, atac, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ torch.log(var.pow(2)) - mean.pow(2) - var.pow(2))

    return reproduction_loss_rna + reproduction_loss_atac + KLD