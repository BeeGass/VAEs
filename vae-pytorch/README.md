<h1 align="center">
  <b>Vanilla VAE</b><br>
</h1>

Here you will find various different VAE implementations done within pytorch, however because this repo focuses on reproducibility, which has proved to be inversely correlated to its readability, I hoped to create a place where future learners may go to and find clear/simple code that helps in their understanding of variational autoencoders. 

In the future, as I implement more I hope to move these READMEs into a directory of their own, until then what is shown below is an implementation of a vanilla VAE. 

## Encoder
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_vector_dim, sub_dim, encoder_type='vanilla'):
        super(Encoder, self).__init__()
        
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
        
        self.read_mu = nn.Linear(latent_vector_dim, sub_dim) # layer to solve for mu, from z
        self.read_log_var = nn.Linear(latent_vector_dim, sub_dim) # layer to solve for sigma, from z

    def forward(self, x):
        latent_vector_z = self.encoder(x)
        mu = self.read_mu(latent_vector_z)
        log_var = self.read_log_var(latent_vector_z)
        return mu, log_var
```

## Decoder
```python
class Decoder(nn.Module):
    def __init__(self, latent_vector_dim, sub_dim, decoder_type='vanilla'): # takes in z
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
```

## VAE 
```python
class VAE(nn.Module):
    def __init__(self, latent_vector_dim, sub_dim, train_bool=True, encoder_type='vanilla', decoder_type='vanilla'):
        super(VAE, self).__init__()
        self._encoder = Encoder(latent_vector_dim, sub_dim, encoder_type)
        self._decoder = Decoder(latent_vector_dim, sub_dim, decoder_type)
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
        log_likeliness = self.log_likelihood(x_hat, x) # also recon_loss
        return log_likeliness + (kl * beta)
```
