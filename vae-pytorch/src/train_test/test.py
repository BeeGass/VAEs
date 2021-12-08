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
        if log_metrics and i % 1000 == 0:
            batch_img = img.reshape(-1, 28, 28)[0].cpu().detach().numpy()
            batch_x_hat = x_hat.reshape(-1, 28, 28)[0].cpu().detach().numpy()
            wandb_img = wandb.Image(batch_img, caption="Top: Output, Bottom: Input")
            wandb_x_hat = wandb.Image(batch_x_hat, caption="Top: Output, Bottom: Input")
            wandb.log({"te-reconstructed image": wandb_x_hat})
            wandb.log({"te-true image": wandb_img})
            
    return loss