import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4):
def train_batch(model, optimizer, train_loader, device, log_metrics):
    torch.manual_seed(42)
    model.train_bool = True
    loss = 1e15
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        img = img.reshape(-1, 28 * 28)
        x_hat, mu, log_var = model(img)
        loss = model.elbo_loss(x_hat, img, mu, log_var)
        if log_metrics:
            wandb.log({"train_loss": loss})
            wandb.log({"tr-reconstructed image": x_hat})
            wandb.log({"tr-true image": img})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return loss
       