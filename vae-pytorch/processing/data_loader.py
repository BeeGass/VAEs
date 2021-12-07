import torch 
from torchvision import datasets
from torchvision import transforms

class MNIST_Dataset():
    def __init__():
        # Transforms images to a PyTorch Tensor
        tensor_transform = transforms.ToTensor()
        
        # Download the MNIST Dataset
        trainset = datasets.MNIST(
            root = './data',
            train = True, 
            download = True,
            transform = tensor_transform
        )
        
        testset = datasets.MNIST(
            root = './data',
            train = False,
            download = True,
            transform = tensor_transform
        )