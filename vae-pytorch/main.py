import wandb
import hydra
import math
import sys
import argparse
import torch
import torch.optim as optim
from src.vae.vae import VAE
from src.train_test.train import train_batch
from src.train_test.test import test_batch
from src.processing.data_process import prepare_datasets, load_datasets
from omegaconf import DictConfig, OmegaConf
  
@hydra.main(config_path="src/conf", config_name="config")
def config_run(cfg : DictConfig) -> None:
    with wandb.init(project="BeeGass-VAE", entity="beegass", config=cfg): # initialize wandb project for logging
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
        
        print(cfg["parameters"]["hidden_dim"])
        model = VAE(latent_vector_dim=cfg["parameters"]["hidden_dim"], 
                    sub_dim=cfg["parameters"]["hidden_sub_dim"], 
                    encoder_type=cfg["models"]["encoder"],
                    decoder_type=cfg["models"]["decoder"]).to(device) # initialize model
        
        trainset, testset = prepare_datasets()
        train_loader, test_loader = load_datasets(trainset, 
                                                  testset, 
                                                  cfg["parameters"]["batch_size"],
                                                  cfg["parameters"]["num_workers"])
        
        optimizer = build_optimizer(model, 
                                    optimizer_name=cfg['parameters']['optimizer'],
                                    learning_rate=cfg['parameters']['learning_rate'],
                                    weight_decay=cfg['parameters']['weight_decay'])
        
        sched = build_scheduler(optimizer=optimizer,
                                sched_name=cfg['parameters']['scheduler'])
        
        train(model=model, 
              optimizer=optimizer, 
              sched=sched, 
              train_loader=train_loader, 
              test_loader=test_loader, 
              device=device, 
              num_epochs=cfg["parameters"]["num_epochs"],
              test_bool=cfg["models"]["test_bool"], 
              log_metrics=cfg["wandb"]["log_metrics"], 
              watch_loss=cfg["wandb"]["watch_loss"])
        
        #torch.onnx.export(model, images, "model.onnx")
        #wandb.save("model.onnx")
        
def tuner() -> None:
    hydra.initialize(config_path="src/conf")
    cfg = hydra.compose(config_name="sweep_config")
    cfg = OmegaConf.to_container(cfg, resolve=[True|False])
    print(type(cfg))
    sweep_id = wandb.sweep(cfg, project="BeeGass-VAE-tune", entity="beegass")
    wandb.agent(sweep_id, function=config_tune)

def config_tune() -> None:
    cfg = {
        'method': 'random', #grid, random
        'metric': {
            'name': 'elbo_loss',
            'goal': 'maximize'   
        },
        'parameters': {
            'optimizer': 'adam',
            'scheduler': 'reduce_lr',
            'hidden_dim': 50,
            'hidden_sub_dim': 30,
            'num_epochs': 100,
            'batch_size': 10,
            'weight_decay': 0.001,
            'learning_rate': 0.001,
            'num_workers': 22,
            'beta': 1
        }
    }
    
    cfg2 = {
        'models': {
            'encoder': 'vanilla', # choices=['vanilla', 'conv', 'resnet']
            'decoder': 'vanilla', # choices=['vanilla', 'conv', 'resnet']
            'test_bool': True
        },
        'wandb': {
            'log_metrics': True,
            'watch_loss': True,
            'tune': False,
        }
    }
    
    with wandb.init(config=cfg):  # defaults are over-ridden during the sweep
        cfg = wandb.config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
        
        model = VAE(latent_vector_dim=cfg["parameters"]["hidden_dim"], 
                    sub_dim=cfg["parameters"]["hidden_sub_dim"], 
                    encoder_type=cfg2["models"]["encoder"],
                    decoder_type=cfg2["models"]["decoder"]).to(device) # initialize model
        
        trainset, testset = prepare_datasets()
        train_loader, test_loader = load_datasets(trainset, 
                                                  testset, 
                                                  cfg["parameters"]["batch_size"],
                                                  cfg["parameters"]["num_workers"])
        
        optimizer = build_optimizer(model, 
                                    optimizer_name=cfg['parameters']['optimizer'],
                                    learning_rate=cfg['parameters']['learning_rate'],
                                    weight_decay=cfg['parameters']['weight_decay'])
        
        sched = build_scheduler(optimizer=optimizer,
                                sched_name=cfg['parameters']['scheduler'])
        
        print("training")
        train(model=model, 
              optimizer=optimizer, 
              sched=sched, 
              train_loader=train_loader, 
              test_loader=test_loader, 
              device=device, 
              num_epochs=cfg["parameters"]["num_epochs"],
              test_bool=cfg2["models"]["test_bool"], 
              log_metrics=cfg2["wandb"]["log_metrics"], 
              watch_loss=cfg2["wandb"]["watch_loss"])
        
        #torch.onnx.export(model, images, "model.onnx")
        #wandb.save("model.onnx")
        
        
def build_optimizer(model, optimizer_name='adam', learning_rate=0.01, weight_decay=0.01, momentum=0.9):
    try:
        optimizer = None
        if optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), 
                                  lr=learning_rate, 
                                  momentum=momentum)
            
        elif optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
               
        return optimizer
    except:
        print("Error: Invalid optimizer specified.")
        sys.exit(1)
        
def build_scheduler(optimizer, sched_name='reduce_lr', patience=5, verbose=True):
    try: 
        sched = None
        if sched_name == "reduce_lr":
            sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         patience=patience, 
                                                         verbose=verbose)
        elif sched_name == 'TODO':
            pass
            #TODO: add other scheduler
            
        return sched
    except:
        print("Error: Invalid scheduler specified.")
        sys.exit(1)

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
            wandb.log({"epoch": epoch})
            wandb.log({"train_loss": train_loss})
        # print(f"Epoch: {epoch+1} \nTrain Loss: {train_loss}")
        sched.step(train_loss)
        
    if test_bool:
        test_loss = test_batch(model, test_loader, device, log_metrics)
        print(f"Test Loss: {test_loss}")



if __name__ == "__main__":
    print("beginning experiment")
    tune_bool = 0
    parser = argparse.ArgumentParser(description='Enable tuning')
    parser.add_argument('--tune', dest='tune', type=int, help='Enable tuning, default is set to 0 (False), omit flag if you wish to run without tuning')
    args = parser.parse_args()
    tune_bool = args.tune
    
    wandb.login()
    if tune_bool:
        tuner()
    else:
        config_run()
        
        
    