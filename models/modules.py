import sys
sys.path.append(".")
import torch
import torch.nn as nn
from torch.distributions import Normal


from dataset.dataloader import PrepareDataloader
from config import Config


def reparameterize_gaussian(mu, var):
    
    return Normal(mu, var.sqrt()).rsample()


class Encoder_rna(nn.Module):
    def __init__(self, input_dim, embed_dim=512, num_heads=8, n_hidden=128, dropout_rate=0.1, n_latent=20):
        super(Encoder_rna, self).__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim, eps=0.0001),
            nn.BatchNorm1d(embed_dim, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.multiheadAttention = nn.MultiheadAttention(embed_dim, num_heads)

        self.MLP2 = nn.Sequential(
            nn.Linear(embed_dim, n_hidden),
            nn.LayerNorm(n_hidden, eps=0.0001),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(n_hidden, n_latent),
            nn.LeakyReLU()
        )

        self.mean_encoder = nn.Linear(n_latent, n_latent)
        self.var_encoder = nn.Linear(n_latent, n_latent)

    def forward(self, x):
        output = self.MLP1(x)
        attn_output, attn_output_weights = self.multiheadAttention(output, output, output)
        output = self.MLP2(attn_output)
        mean = self.mean_encoder(output)
        var = self.var_encoder(output)
        return output, mean, var
    

class Encoder_atac(nn.Module):
    def __init__(self, input_dim, embed_dim=512, num_heads=8, n_hidden=128, dropout_rate=0.1, n_latent=20):
        super(Encoder_atac, self).__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim, eps=0.0001),
            nn.BatchNorm1d(embed_dim, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.multiheadAttention = nn.MultiheadAttention(embed_dim, num_heads)

        self.MLP2 = nn.Sequential(
            nn.Linear(embed_dim, n_hidden),
            nn.LayerNorm(n_hidden, eps=0.0001),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate), 

            nn.Linear(n_hidden, n_latent),
            nn.LeakyReLU()
        )

        self.mean_encoder = nn.Linear(n_latent, n_latent)
        self.var_encoder = nn.Linear(n_latent, n_latent)

    def forward(self, x):
        output = self.MLP1(x)
        attn_output, attn_output_weights = self.multiheadAttention(output, output, output)
        output = self.MLP2(attn_output)
        mean = self.mean_encoder(output)
        var = self.var_encoder(output)

        return output, mean, var
    
class Encoder(nn.Module):
    def __init__(self, rna_input_dim, atac_input_dim, rna_embed_dim=512, rna_num_heads=8, rna_n_hidden=128, rna_dropout_rate=0.1,  
                 atac_embed_dim=512, atac_num_heads=8, atac_n_hidden=128, atac_dropout_rate=0.1, latent_dim=20):
        super(Encoder, self).__init__()
        self.encoder_rna = Encoder_rna(rna_input_dim, rna_embed_dim, rna_num_heads, 
                                       rna_n_hidden, rna_dropout_rate, latent_dim)
        self.encoder_atac = Encoder_atac(atac_input_dim, atac_embed_dim, atac_num_heads, 
                                         atac_n_hidden, atac_dropout_rate, latent_dim)
        self.MLP = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU()
        )
        self.mean_encoder = nn.Linear(latent_dim, latent_dim)
        self.var_encoder = nn.Linear(latent_dim, latent_dim)

    def forward(self, rna, atac):
        rna_output, rna_mean, rna_var = self.encoder_rna(rna)
        atac_output, atac_mean, atac_var = self.encoder_atac(atac)
        rna_atac = torch.concatenate((rna_output, atac_output), dim=1)
        output = self.MLP(rna_atac)
        
        # calculate mean and variance
        mean = self.mean_encoder(output)
        var = self.var_encoder(output)
        
        
        return mean, var

class Decoder_rna(nn.Module):
    def __init__(self, input_dim, embed_dim=512, num_heads=8, n_hidden=128, dropout_rate=0.1, n_latent=20) -> None:
        super(Decoder_rna, self).__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.LeakyReLU(),

            nn.Linear(n_hidden, embed_dim),
            nn.LayerNorm(embed_dim, eps=0.0001),
            nn.BatchNorm1d(embed_dim, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.multiheadAttention = nn.MultiheadAttention(embed_dim, num_heads)
        self.MLP2 = nn.Sequential(
            nn.Linear(embed_dim, input_dim),
            nn.Tanh()
            # nn.Sigmoid()
        )
    
    def forward(self, latent):
        output = self.MLP1(latent)
        attn_output, attn_output_weights = self.multiheadAttention(output, output, output)
        output = self.MLP2(attn_output)
        return output


class Decoder_atac(nn.Module):
    def __init__(self, input_dim, embed_dim=512, num_heads=8, n_hidden=128, dropout_rate=0.1, n_latent=20) -> None:
        super(Decoder_atac, self).__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.LeakyReLU(),

            nn.Linear(n_hidden, embed_dim),
            nn.LayerNorm(embed_dim, eps=0.0001),
            nn.BatchNorm1d(embed_dim, momentum=0.01, eps=0.0001),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.multiheadAttention = nn.MultiheadAttention(embed_dim, num_heads)
        self.MLP2 = nn.Sequential(
            nn.Linear(embed_dim, input_dim),
            nn.Tanh()
            # nn.Sigmoid()
        )
    
    def forward(self, latent):
        output = self.MLP1(latent)
        attn_output, attn_output_weights = self.multiheadAttention(output, output, output)
        output = self.MLP2(attn_output)
        return output

class Decoder(nn.Module):
    def __init__(self, rna_input_dim, atac_input_dim, rna_embed_dim=512, rna_num_heads=8, rna_n_hidden=128, rna_dropout_rate=0.1,  
                 atac_embed_dim=512, atac_num_heads=8, atac_n_hidden=128, atac_dropout_rate=0.1, latent_dim=20):
        super(Decoder, self).__init__()
        self.decoder_rna = Decoder_rna(rna_input_dim, rna_embed_dim, rna_num_heads, 
                                       rna_n_hidden, rna_dropout_rate, latent_dim)
        self.decoder_atac = Decoder_atac(atac_input_dim, atac_embed_dim, atac_num_heads, 
                                         atac_n_hidden, atac_dropout_rate, latent_dim)
    
    def forward(self, latent):
        rna_generated = self.decoder_rna(latent)
        atac_generated = self.decoder_atac(latent)
        
        return rna_generated, atac_generated


if __name__ == "__main__":
    config = Config()
    train_loader, test_loader, rna_input_size, atac_input_size, num_of_batch = PrepareDataloader(config).getloader()

    encoder = Decoder(rna_input_size, atac_input_size).to(config.device).double()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = torch.nn.DataParallel(encoder, device_ids=config.device_ids)
        encoder = encoder.double()

    for step, (x, y) in enumerate(train_loader):
        x = x.reshape(-1, rna_input_size).to(config.device)
        y = y.reshape(-1, atac_input_size).to(config.device)
        # x = x.to(config.device)
        # y = y.to(config.device)
        latent, rna_generated, atac_generated = encoder(x, y)
        print(latent)
        print(rna_generated)
        print(atac_generated)
        # break    