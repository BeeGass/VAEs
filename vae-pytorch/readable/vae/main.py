"""
# Auto-Encoding Variational Bayes - VAE in PyTorch
This is a bare bones implementation of a Variational Autoencoder as presented by the [paper](https://arxiv.org/abs/1312.6114). 

This implementation is trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), **the code for this tutorial is available at
[Github BeeGass/Readable-VAEs](https://github.com/BeeGass/Readable-VAEs/blob/master/vae-pytorch/readable/vae/main.py).** . 

If someone reading this has any questions or comments please find me on Twitter, [@BeeAGass](https://twitter.com/BeeAGass).

*disclaimer*: VAEs are a wonderful step into the world of generative models, however many aspects of them simply cannot be covered in a tutorial like this because it would simply take to long. 
This tutorial is designed to give a high level understanding into the world of VAEs.
"""
import torch
import torch.nn as nn

"""
## Idea Overview:
VAEs can be thought of as combination of three components.
1. Encoder
2. Decoder
3. Manipulation of the data inbetween the encoder and the decoder that we know as the **latent space**. This will be covered as you read further

### Encoder
The basic idea behind encoders is that we wish to compress our data $x$. 
The encoder networks learns to take in our input data and compress it such that the ouput is a feature rich representation of the input. 
The output of our encoder is $\mu$ and $\sigma$, this is our latent space 

### Decoder
The basic idea behind decoders is that we wish to take a sample of our compressed data and map it back up into, a higher dimensionality, a dimensionality equivalent to our input data. 
The output of our decoder is $\hat{x}$, this is our reconstructed data

### Latent Space Manipulation
We know that the encoder compresses each data point, $x$ into a feature rich representation that we have denoted as `z`. 
`z` is, in our case, a gaussian probability distribution which gets broken into its components of $\mu$ and $\sigma$, that represent the mean and variance, which describe our distribution. 

All of these terms and concepts will be elaborated on more as you read further. 
"""
class Encoder(nn.Module):
    def __init__(self, latent_dims): # `latent_dims`: The number associated with dimensionality you wish to compress your data to
        super().__init__()

        """
        Here we initialize the encoder to accept a dimensionality equal to a 28x28 pixel image, our input image from the MNIST dataset.
        The architecture should be one that has descending layers. In this case we are eventually compressing our image by $\approx$15 times
        """
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Linear(256, latent_dim)
        )
        self.read_mu = nn.Linear(latent_dim, latent_dims/2) # layer to solve for mu, from z
        self.read_log_var = nn.Linear(latent_dim, latent_dims/2) # layer to solve for sigma, from z

    def forward(self, x):
        latent_space_z = self.encoder(x) # obtain latent space `z`
        mu = self.read_mu(latent_space_z) # decompose latent space into $\mu$, mean
        log_var = self.read_log_var(latent_space_z) # decompose latent space into $\sigma$, variance
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dims, latent_comp_dim):
        super().__init__()

        """
        Here we initialize the decoder to accept a shape equal to the shape both latent vector components were set to. 
        The architecture should be one that has ascending layers that maps up in the same way that the encoder maps down, in this case we map up to a 28x28 pixel image. 
        """
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim/2, latent_dim)
            nn.Linear(latent_dim, 256)
            nn.Linear(256, 784)
        )

    def forward(self, z):
        return self.decoder(z) # The output is $\hat{x}$


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._encoder = Encoder(latent_dim) # initialize encoder
        self.decoder = Decoder(latent_dim) # initialize decoder

    def forward(self, x):
        mu, log_var = self._encoder(x) # obtain `mu` and `log_var` aka the latent space from input data
        z = self.reparameterize(mu, log_var) # perform reparameterization trick to sample from the probability distribution
        x_hat = self._decoder(z) # input sample into decoder use it to create our reconstructed image 
        return x_hat, mu, log_var 

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # $\text{standard deviation} = e^{\frac{1}{2} \log \sigma} = e^{\log \sigma^{2}}$
        epsilon = torch.randn_like(std) # randomly pick numbers (between 0 and 1) from a matrix of the same shape as the standard deviation matrix 
        sigma = epsilon * std # multiply the random value by our std matrix to obtain our variance
        return mu + sigma # adding $\mu$ to our $\sigma$ gives us the values of our sampled sythetic distribution 
    
    def elbo_loss(self, x_hat, x, mu, log_var, beta=1):
        """
        ## ELBO Loss
        The general form:
        $$\E_{q(z|x, \Theta)} \log p(x|z, $$ TODO
         """
        kl = self.kl_divergence(mu, log_var)
        log_likeliness = self.log_likelihood(x_hat, x) # also recon_loss
        return log_likeliness + (kl * beta)

    def kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # *for the log_var.exp() portion*: e^ln(var) = var = sigma^2

    def log_likelihood(self, x_hat, x):
        return torch.mean(torch.pow(x_hat - x, 2)) # log_likelihood of x_hat of the data under the model

LATENT_DIM = 0
LATENT_COMP_DIM = 0
BATCH_SIZE = 0
NUM_WORKERS = 0
LEARNING_RATE = 0
WEIGHT_DECAY = 0

def train_batch(model, optimizer, train_loader, device, log_metrics):
    torch.manual_seed(42)
    model.train_bool = True
    loss = 1e15
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        img = img.reshape(-1, 28 * 28)
        x_hat, mu, log_var = model(img)
        loss = model.elbo_loss(x_hat, img, mu, log_var)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss

def test_batch(model, test_loader, device, log_metrics):
    torch.manual_seed(42)
    loss = 1e15
    for data in test_loader:
        img, _ = data
        img = img.to(device)
        img = img.reshape(-1, 28 * 28)
        x_hat, mu, log_var = model(img)
        loss = model.elbo_loss(x_hat, img, mu, log_var)
            
    return loss

def train(model, optimizer, sched, train_loader, test_loader, device, num_epochs=100, test_bool=True, log_metrics=False, watch_loss=False):
    for epoch in range(num_epochs):
        train_loss = train_batch(model, optimizer, train_loader, device, log_metrics)
        sched.step(train_loss)

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
    model = VAE(latent_dim=latent_dim, latent_comp_dim=latent_comp_dim)
    trainset, testset = prepare_datasets()
    train_loader, test_loader = load_datasets(trainset, testset, batch_size, num_workers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

if __name__ == "__main__":

