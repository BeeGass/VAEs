import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dims, latent_comp_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Linear(256, latent_dim)
        )

        self.read_mu = nn.Linear(latent_dim, latent_comp_dim) # layer to solve for mu, from z
        self.read_log_var = nn.Linear(latent_dim, latent_comp_dim) # layer to solve for sigma, from z

    def forward(self, x):
        latent_vector_z = self.encoder(x)
        mu = self.read_mu(latent_vector_z)
        log_var = self.read_log_var(latent_vector_z)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dims, latent_comp_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_comp_dim, latent_dim)
            nn.Linear(latent_dim, 256)
            nn.Linear(256, 784)
        )

    def forward(self, x): # outputs x
        return self.decoder(x)


class VAE(nn.Module):
    def __init__(self, latent_dim, latent_comp_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim, latent_comp_dim)
        self.decoder = Decoder(latent_dim, latent_comp_dim)

    def forward(self, x):
        mu, log_var = self._encoder(x) 
        z = self.reparameterize(mu, log_var) 
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

