import torch
import numpy as np
import torch.nn.functional as F

class Losses():
    def __init__():
        super(Losses, self).__init__()
    
    def kl_divergence(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # *for the log_var.exp() portion*: e^ln(var) = var = sigma^2

    def log_likelihood(x_hat, x):
        return torch.mean(torch.pow(x_hat - x, 2)) # log_likelihood of x_hat of the data under the model

    def elbo(x_hat, x, mu, log_var, beta=1):
        kl = kl_divergence(mu, log_var)
        # recon_loss = F.binary_cross_entropy(x_hat.view(-1, 784), x.view(-1, 784), reduction='sum')
        log_likeliness = log_likelihood(x_hat, x) # also recon_loss
        return log_likeliness + (kl * beta)