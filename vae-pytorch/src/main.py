from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
import torch
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from vae import VAE
from loss import Losses
from train import train
from test import test


def prepare_datasets():
    # Transforms images to a PyTorch Tensor
    tensor_transform = transforms.ToTensor()
    
    # Download the MNIST Dataset
    trainset = datasets.MNIST(
        root = './vae-pytorch/data',
        train = True, 
        download = True,
        transform = tensor_transform
    )
    
    testset = datasets.MNIST(
        root = './vae-pytorch/data',
        train = False,
        download = True,
        transform = tensor_transform
    )
    
    return trainset, testset

def load_datasets(trainset, testset, batch_size, num_workers):
    # DataLoader is used to load the dataset 
    # for training
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = num_workers
    )
    
    # DataLoader is used to load the dataset 
    # for testing
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = num_workers
    )
    
    return train_loader, test_loader
  
@hydra.main(config_path="conf", config_name="config")
def config_run(cfg : DictConfig) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
    
    trainset, testset = prepare_datasets()
    train_loader, test_loader = load_datasets(trainset, testset, cfg["train"]["batch_size"], cfg["train"]["num_workers"])
    
    optimizer = optim.Adam(model.parameters(), 
                       lr=cfg["train"]["learning_rate"], 
                       weight_decay=cfg["train"]["weight_decay"])
    
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode='min', 
                                                 patience=5, 
                                                 verbose=True)
    
    model = VAE(latent_vector_dim=cfg["models"]["hidden_dim"], sub_dim=cfg["models"]["hidden_sub_dim"]).to(device) # initialize model
    
    run(model, optimizer, sched, train_loader, test_loader, cfg["train"]["num_epochs"], device)

def run(model, optim, sched, train_loader, test_loader, num_epochs=100, device):
    for epoch in range(num_epochs):
        train_loss = train(model, optim, sched, train_loader, device, watch_loss=False)
        test_loss = test(model, test_loader, device, watch_loss=False)
        if log_metrics:
            wandb.log({"loss": loss})
            wandb.log({"epoch": epoch})
            wandb.log({"reconstructed image": x_hat})
            wandb.log({"true image": img})
            wandb.log({"lr": scheduler.get_lr()})
            wandb.log({"train_loss": train_loss})
            wandb.log({"test_loss": test_loss})
        scheduler.step(test_loss)



if __name__ == "__main__":
    wandb.init(project="BeeGass-VAE", entity="beegass") # initialize wandb project for logging
    config_run()
        
    