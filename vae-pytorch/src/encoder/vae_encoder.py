import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_vector_dim, sub_dim, encoder_type='vanilla'):
        super(Encoder, self).__init__()
        
        if encoder_type == 'vanilla':
            # architecture to solve for latent vector, z
            self.encoder = nn.Sequential(
                nn.Linear(28*28, 392),
                nn.ReLU(),
                nn.Linear(392, 196),
                nn.ReLU(),
                nn.Linear(196, 98),
                nn.ReLU(),
                nn.Linear(98, 48),
                nn.ReLU(),
                nn.Linear(48, latent_vector_dim),
                nn.ReLU()
                
            )
        elif encoder_type == 'conv':
            #TODO: implement convolutional encoder
            pass
        
        elif encoder_type == 'resnet':
            #TODO: implement resnet encoder
            pass
        
        self.read_mu = nn.Linear(latent_vector_dim, sub_dim) # layer to solve for mu, from z
        self.read_log_var = nn.Linear(latent_vector_dim, sub_dim) # layer to solve for sigma, from z

    def forward(self, x):
        latent_vector_z = self.encoder(x)
        mu = self.read_mu(latent_vector_z)
        log_var = self.read_log_var(latent_vector_z)
        return mu, log_var