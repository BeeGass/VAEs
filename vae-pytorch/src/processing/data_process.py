import torch 
from torchvision import datasets
from torchvision import transforms

def prepare_datasets():
    # Transforms images to a PyTorch Tensor
    tensor_transform = transforms.ToTensor()
    
    # Download the MNIST Dataset
    trainset = datasets.MNIST(
        root = './vae-pytorch/src/data',
        train = True, 
        download = True,
        transform = tensor_transform
    )
    
    testset = datasets.MNIST(
        root = './vae-pytorch/src/data',
        train = False,
        download = True,
        transform = tensor_transform
    )
    
    return trainset, testset

def load_datasets(trainset, testset, batch_size, num_workers):
    # DataLoader is used to load the dataset 
    # for training
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = num_workers
    )
    
    # DataLoader is used to load the dataset 
    # for testing
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = num_workers
    )
    
    return train_loader, test_loader