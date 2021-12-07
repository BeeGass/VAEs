import torch

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