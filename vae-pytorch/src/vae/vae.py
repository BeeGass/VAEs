import numpy as np
import torch
import torch.nn as nn
from vae_encoder import Encoder
from vae_decoder import Decoder

class VAE(nn.Module):
    def __init__(self, latent_vector_dim, sub_dim, train_bool=True):
        super(VAE, self).__init__()
        self._encoder = Encoder(latent_vector_dim, sub_dim)
        self._decoder = Decoder(latent_vector_dim, sub_dim)
        self.train_bool = train_bool

    def forward(self, x):
        mu, log_var = self._encoder(x) 
        if self.train_bool:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu 
            
        x_hat = self._decoder(z)       
        return x_hat, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        sigma = epsilon * std
        return mu + sigma
    
    def kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # *for the log_var.exp() portion*: e^ln(var) = var = sigma^2

    def log_likelihood(self, x_hat, x):
        return torch.mean(torch.pow(x_hat - x, 2)) # log_likelihood of x_hat of the data under the model

    def elbo_loss(self, x_hat, x, mu, log_var, beta=1):
        kl = self.kl_divergence(mu, log_var)
        # recon_loss = F.binary_cross_entropy(x_hat.view(-1, 784), x.view(-1, 784), reduction='sum')
        log_likeliness = self.log_likelihood(x_hat, x) # also recon_loss
        return log_likeliness + (kl * beta)