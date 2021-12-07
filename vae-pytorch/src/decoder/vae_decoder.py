import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_vector_dim, sub_dim, decoder_type='vanilla'): # takes in z
        super(Decoder, self).__init__()
        
        if decoder_type == 'vanilla':
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
        elif decoder_type == 'conv':
            #TODO: implement convolutional encoder
            pass
        
        elif decoder_type == 'resnet':
            #TODO: implement resnet encoder
            pass
        

    def forward(self, x): # outputs x
        return self.decoder(x)