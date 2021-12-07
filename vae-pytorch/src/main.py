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
from data_process import prepare_datasets, load_datasets

  
@hydra.main(config_path="./vae-pytorch/src/conf", config_name="config")
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
    
    model = VAE(latent_vector_dim=cfg["models"]["hidden_dim"], 
                sub_dim=cfg["models"]["hidden_sub_dim"], 
                encoder_type=cfg["models"]["encoder"],
                decoder_type=cfg["models"]["decoder"]).to(device) # initialize model
    
    run(model, optimizer, sched, train_loader, test_loader, device, cfg["train"]["num_epochs"])

def run(model, optim, sched, train_loader, test_loader, device, num_epochs=100):
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
        
    