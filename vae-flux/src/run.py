from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
from models.vae import VAE
from loss import Losses

@hydra.main(config_path="conf", config_name="config")
def train_model(cfg : DictConfig) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
    loss_fun = Losses().elbo # loss function
    model = VAE(latent_vector_dim=latent_vector_dim, sub_dim=sub_dim, train_bool=True).to(device) # initialize model
    return train(model, loss_fun, num_epochs, batch_size, learning_rate, wd, watch_loss, log_metrics) # train model
    
@hydra.main(config_path="conf", config_name="config")
def run(cfg : DictConfig) -> None:
    trained_model = train_model()
    if cfg.train.test:
        trained_model.train_bool = False
        test(trained_model, loss_fun, num_epochs, batch_size, learning_rate, wd, watch_loss, log_metrics) # test model if trained
        
        
class Experiment():
    def __init__(self, ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
        loss_fun = Losses().elbo # loss function
        

if __name__ == "__main__":
    wandb.init(project="BeeGass-VAE", entity="beegass") # initialize wandb project for logging
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
    loss_fun = Losses().elbo # loss function
    train_model() # train model
        
    