import sys
sys.path.append(".")
import torch
import torch.nn as nn
from torch.distributions import Normal
from dataset.dataloader import PrepareDataloader
from config import Config


from modules import Encoder, Decoder

# torch.set_default_dtype(torch.float64)

class UnnamedModel(nn.Module):
    def __init__(self, rna_input_dim, atac_input_dim, rna_embed_dim=512, rna_num_heads=8, rna_n_hidden=128, rna_dropout_rate=0.1,  
                 atac_embed_dim=512, atac_num_heads=8, atac_n_hidden=128, atac_dropout_rate=0.1, latent_dim=20):
        super(UnnamedModel, self).__init__()
        self.encoder = Encoder(rna_input_dim, atac_input_dim)
        self.decoder = Decoder(rna_input_dim, atac_input_dim)
    
    def reparameterize_gaussian(self, mu, var):
        
        return Normal(mu, var.sqrt()).rsample()
    
    def forward(self, rna, atac):
        mean, var = self.encoder(rna, atac)

        latent = self.reparameterize_gaussian(mean, var)

        rna_generated, atac_generated = self.decoder(latent)
        
        return latent, rna_generated, atac_generated



if __name__ == "__main__":
    config = Config()
    train_loader, test_loader, rna_input_size, atac_input_size, num_of_batch = PrepareDataloader(config).getloader()

    model = UnnamedModel(rna_input_size, atac_input_size).to(config.device).double()
    
    print(f"the number of batch is {num_of_batch}")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    for step, (x, y) in enumerate(train_loader):
        print(step)
        x = x.reshape(-1, rna_input_size).to(config.device)
        y = y.reshape(-1, atac_input_size).to(config.device)
        # x = x.to(config.device)
        # y = y.to(config.device)
        latent, rna_generated, atac_generated = model(x, y)
        # print(latent)
        # print(rna_generated)
        # print(atac_generated)
        # break    