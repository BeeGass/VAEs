import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb


def test_batch(model, test_loader, device, log_metrics):
    torch.manual_seed(42)
    model.train_bool = False
    loss = 1e15
    for data in test_loader:
        img, _ = data
        img = img.to(device)
        img = img.reshape(-1, 28 * 28)
        x_hat, mu, log_var = model(img)
        loss = model.elbo_loss(x_hat, img, mu, log_var)
        if log_metrics:
            wandb.log({"test_loss": loss})
            wandb.log({"te-reconstructed image": x_hat})
            wandb.log({"te-true image": img})
            
    return loss