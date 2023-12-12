import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from dataset.dataloader import PrepareDataloader
from config import Config
from models.model import UnnamedModel
from models.loss import loss_function


def main():
    config = Config()
    train_loader, test_loader, rna_input_size, atac_input_size, num_of_batch = PrepareDataloader(config).getloader()

    model = UnnamedModel(rna_input_size, atac_input_size).to(config.device).double()
    
    print(f"the number of batch is {num_of_batch}")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(config.epochs):
        overall_loss = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.reshape(-1, rna_input_size).to(config.device)
            y = y.reshape(-1, atac_input_size).to(config.device)

            optimizer.zero_grad()

            latent, rna_generated, atac_generated, mean, var= model(x, y)
            
            loss = loss_function(x, rna_generated, mean, var)
            loss += loss_function(y, atac_generated, mean, var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(step * config.batch_size))    


if __name__ == "__main__":
    main()