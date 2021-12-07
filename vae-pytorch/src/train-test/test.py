import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4):
def test(model, test_loader, device, watch_loss=False):
    torch.manual_seed(42)
    model.train_bool = False
    loss = 1e15
    for data in test_loader:
        img, _ = data
        img = img.to(device)
        img = img.reshape(-1, 28 * 28)
        x_hat, mu, log_var = model(img)
        loss = model.elbo_loss(x_hat, img, mu, log_var)
        if watch_loss:
            wandb.watch(model)
            
    return loss