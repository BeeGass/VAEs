import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4):
def train(model, optim, sched, train_loader, device, watch_loss=False):
    torch.manual_seed(42)
    model.train_bool = True
    loss = 1e15
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        img = img.reshape(-1, 28 * 28)
        x_hat, mu, log_var = model(img)
        loss = model.elbo_loss(x_hat, img, mu, log_var)
        loss.backward()
        optimizer.step()
        if watch_loss:
            wandb.watch(model)
        optimizer.zero_grad()
        
    return loss
       