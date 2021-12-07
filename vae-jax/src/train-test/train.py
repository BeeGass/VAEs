import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4):
def train(model, loss_fun, num_epochs=5, batch_size=28, learning_rate=1e-4, wd=1e-5, watch_loss=False, log_metrics=False):
    torch.manual_seed(42)
    optimizer = optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=wd) # <--
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min',
                                                     patience=5, 
                                                     verbose=True)
    
    # DataLoader is used to load the dataset 
    # for training
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = 22
    )
    
    loss = 1e15
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            img = img.reshape(-1, 28 * 28)
            x_hat, mu, log_var = model(img)
            loss = loss_fun(x_hat, img, mu, log_var)
            loss.backward()
            optimizer.step()
            if watch_loss:
                wandb.watch(model)
            optimizer.zero_grad()
            
        if log_metrics:
            wandb.log({"loss": loss})
            wandb.log({"epoch": epoch})
            wandb.log({"reconstructed image": x_hat})
            wandb.log({"true image": img})
            wandb.log({"lr": scheduler.get_lr()})
        scheduler.step(loss)
        
    return model
       