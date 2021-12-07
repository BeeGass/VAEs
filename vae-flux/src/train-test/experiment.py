import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

class Experiment():
    def __init__(self, model, loss_fun, num_epochs=100, batch_size=28, learning_rate=1e-4, wd=1e-5, seed=42, watch_loss=False, log_metrics=False):
        self.model = model
        self.loss_fun = loss_fun
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.watch_loss = watch_loss
        self.log_metrics = log_metrics
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
        
    # def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4):
    def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4, wd=1e-5, watch_loss=False, log_metrics=False):
        torch.manual_seed(self.seed)
        optimizer = optim.Adam(model.parameters(),
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay) # <--
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min',
                                                        patience=5, 
                                                        verbose=True)
        
        # DataLoader is used to load the dataset 
        # for training
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size = self.batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = 22
        )
        
        loss = 1e15
        for epoch in range(num_epochs):
            for data in train_loader:
                img, _ = data
                img = img.to(self.device)
                img = img.reshape(-1, 28 * 28)
                x_hat, mu, log_var = model(img)
                loss = loss_fun(x_hat, img, mu, log_var)
                loss.backward()
                optimizer.step()
                if watch_loss:
                    wandb.watch(model)
                optimizer.zero_grad()
                
            if log_metrics:
                wandb.log({"loss": loss})
                wandb.log({"epoch": epoch})
                wandb.log({"reconstructed image": x_hat})
                wandb.log({"true image": img})
                wandb.log({"lr": scheduler.get_lr()})
            scheduler.step(loss)
            
        return model