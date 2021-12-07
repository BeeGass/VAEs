import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_vector_dim, sub_dim): # takes in z
        super(Decoder, self).__init__()
        
        # architecture to reconstruct for latent vector, z
        self.decoder = nn.Sequential(
            nn.Linear(sub_dim, latent_vector_dim),
            nn.ReLU(),
            nn.Linear(latent_vector_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 98),
            nn.ReLU(),
            nn.Linear(98, 196),
            nn.ReLU(),
            nn.Linear(196, 392),
            nn.ReLU(),
            nn.Linear(392, 28*28),
            nn.Sigmoid()
        )
        

    def forward(self, x): # outputs x
        return self.decoder(x)