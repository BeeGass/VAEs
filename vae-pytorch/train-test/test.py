import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from data_loader import MNIST_Dataset

testset = MNIST_Dataset().testset

# def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4):
def test(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4, wd=1e-5, watch_loss=False, log_metrics=False):
    torch.manual_seed(42)
    
    # DataLoader is used to load the dataset 
    # for testing
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = 22
    )
    
    loss = 1e15
    for epoch in range(num_epochs):
        for data in test_loader:
            img, _ = data
            img = img.to(device)
            img = img.reshape(-1, 28 * 28)
            x_hat, mu, log_var = model(img)
            loss = loss_fun(x_hat, img, mu, log_var)
            if watch_loss:
                wandb.watch(model)
            
        if log_metrics:
            wandb.log({"loss": loss})
            wandb.log({"epoch": epoch})
            wandb.log({"reconstructed image": x_hat})
            wandb.log({"true image": img})