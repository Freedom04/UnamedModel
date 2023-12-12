import math
import sys
sys.path.append(".")
import torch
import torch.nn as nn
from torch.distributions import Normal
from dataset.dataloader import PrepareDataloader
from config import Config
import numpy as np

# torch.set_default_dtype(torch.float64)

def reparameterize_gaussian(mu, var):
    
    return Normal(mu, var.sqrt()).rsample()


class Encoder_rna(nn.Module):
    def __init__(self, embed_dim, num_heads=8, n_hidden=128, dropout_rate=0.1, n_output=20):
        super(Encoder_rna, self).__init__()
        self.multiheadAttention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, 128)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, n_hidden),
            nn.LayerNorm(n_hidden, eps=0.0001),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(n_hidden, n_output),
            nn.LeakyReLU()
        )
        self.mean_encoder = nn.Linear(n_output, n_output)
        self.var_encoder = nn.Linear(n_output, n_output)

    def forward(self, x):
        attn_output, attn_output_weights = self.multiheadAttention(x, x, x)
        output = self.MLP(attn_output)
        mean = self.mean_encoder(output)
        var = torch.exp(self.var_encoder(output)) + 1e-4
        latent = reparameterize_gaussian(mean, var)
        return output, mean, var, latent
    

class Encoder_atac(nn.Module):
    def __init__(self, embed_dim, num_heads=8, n_hidden=128, dropout_rate=0.1, n_output=20):
        super(Encoder_atac, self).__init__()
        self.multiheadAttention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, 128)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, n_hidden),
            nn.LayerNorm(n_hidden, eps=0.0001),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(n_hidden, n_output),
            nn.LeakyReLU()
        )
        self.mean_encoder = nn.Linear(n_output, n_output)
        self.var_encoder = nn.Linear(n_output, n_output)

    def forward(self, x):
        attn_output, attn_output_weights = self.multiheadAttention(x, x, x)
        output = self.MLP(attn_output)
        mean = self.mean_encoder(output)
        var = torch.exp(self.var_encoder(output)) + 1e-4
        latent = reparameterize_gaussian(mean, var)
        return output, mean, var, latent
    
class Encoder(nn.Module):
    def __init__(self, rna_embed_dim, atac_embed_dim, rna_num_heads=8, rna_n_hidden=128, rna_dropout_rate=0.1,  
                 atac_num_heads=8, atac_n_hidden=128, atac_dropout_rate=0.1, latent_dim=20):
        super(Encoder, self).__init__()
        self.encoder_rna = Encoder_rna(rna_embed_dim, rna_num_heads, rna_n_hidden, rna_dropout_rate, latent_dim)
        self.encoder_atac = Encoder_atac(atac_embed_dim, atac_num_heads, atac_n_hidden, atac_dropout_rate, latent_dim)
        self.MLP = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU()
        )
        self.mean_encoder = nn.Linear(latent_dim, latent_dim)
        self.var_encoder = nn.Linear(latent_dim, latent_dim)

    def forward(self, rna, atac):
        rna_output, rna_mean, rna_var, rna_latent = self.encoder_rna(rna)
        atac_output, atac_mean, atac_var, atac_latent = self.encoder_atac(atac)
        rna_atac = torch.concatenate((rna_output, atac_output), dim=1)
        output = self.MLP(rna_atac)
        
        # calculate mean and variance
        mean = self.mean_encoder(output)
        var = torch.exp(self.var_encoder(output)) + 1e-4
        
        # reparameterize latent
        latent = reparameterize_gaussian(mean, var)
        
        return latent, mean, var




if __name__ == "__main__":
    config = Config()
    train_loader, test_loader, rna_input_size, atac_input_size, num_of_batch = PrepareDataloader(config).getloader()

    encoder = Encoder(rna_input_size, atac_input_size).to(config.device).double()

    for step, (x, y) in enumerate(train_loader):
        x = x.reshape(-1, rna_input_size).to(config.device)
        y = y.reshape(-1, atac_input_size).to(config.device)
        # x = x.to(config.device)
        # y = y.to(config.device)
        latent, mean, var = encoder(x, y)
        print(latent)
        print(mean)
        print(var)
        break    