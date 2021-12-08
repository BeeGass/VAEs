from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
import torch.optim as optim
import torch
from src.vae.vae import VAE
from src.train_test.train import train_batch
from src.train_test.test import test_batch
from src.processing.data_process import prepare_datasets, load_datasets
import sys

# sys.path.insert(0, 'vae-pytorch/src')
  
@hydra.main(config_path="src/conf", config_name="config")
def config_run(cfg : DictConfig) -> None:
    wandb.login()
    with wandb.init(project="BeeGass-VAE", entity="beegass", config=cfg): # initialize wandb project for logging
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
        model = VAE(latent_vector_dim=cfg["models"]["hidden_dim"], 
                    sub_dim=cfg["models"]["hidden_sub_dim"], 
                    encoder_type=cfg["models"]["encoder"],
                    decoder_type=cfg["models"]["decoder"]).to(device) # initialize model
        trainset, testset = prepare_datasets()
        train_loader, test_loader = load_datasets(trainset, 
                                                testset, 
                                                cfg["train"]["batch_size"], 
                                                cfg["train"]["num_workers"])
        
        optimizer = optim.Adam(model.parameters(), 
                            lr=cfg["train"]["learning_rate"], 
                            weight_decay=cfg["train"]["weight_decay"])
        
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode='min', 
                                                    patience=5, 
                                                    verbose=True)
        
        train(model=model, 
            optimizer=optimizer, 
            sched=sched, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            device=device, 
            num_epochs=cfg["train"]["num_epochs"],
            test_bool=cfg["train"]["test"], 
            log_metrics=cfg["wandb"]["log_metrics"], 
            watch_loss=cfg["wandb"]["watch_loss"])
        
        torch.onnx.export(model, images, "model.onnx")
        wandb.save("model.onnx")

def train(model, optimizer, sched, train_loader, test_loader, device, num_epochs=100, test_bool=True, log_metrics=False, watch_loss=False):
    print("beginning run")
    if watch_loss:
            wandb.watch(model,
                        criterion=model.elbo_loss,
                        log_freq=10, 
                        idx=None, 
                        log_graph=False)
    for epoch in range(num_epochs):
        train_loss = train_batch(model, optimizer, train_loader, device, log_metrics)
        if log_metrics:
            wandb.log({"train_loss": train_loss})
            wandb.log({"epoch": epoch})
        print(f"Epoch: {epoch+1} \nTrain Loss: {train_loss}")
        sched.step(train_loss)
        
    if test_bool:
        test_loss = test_batch(model, test_loader, device, log_metrics)
        print(f"Test Loss: {test_loss}")



if __name__ == "__main__":
    print("beginning experiment")
    config_run()
        
    