import torch
from torchvision import datasets, transforms
import numpy as np

class NN(torch.nn.Module):
    """Boilerplate neural network class"""
    def __init__(self, layers, embedding=False, distribution=False, flatten_input=False):
        super().__init__()

        self.flatten_input = flatten_input

        l = []
        for idx in range(len(layers) - 1):
            l.append(torch.nn.Linear(layers[idx], layers[idx+1]))   # add a linear layer
            if idx + 1 != len(layers) - 1: # if this is not the last layer ( +1 = zero indexed) (-1 = layer b4 last)
                l.append(torch.nn.ReLU())   # activate
        if distribution:    # if a probability dist output is required
            l.append(torch.nn.Softmax())    # apply softmax to output
            
        self.layers = torch.nn.Sequential(*l) # unpack layers & turn into a function which applies them sequentially 

    def forward(self, x):
        if self.flatten_input:
            x = x.view(x.shape[0], np.prod(x.shape[1:]))
        return self.layers(x)


def get_splits():

    # GET THE TRAINING DATASET
    train_data = datasets.MNIST(root='MNIST-data',                        # where is the data (going to be) stored
        transform=transforms.ToTensor(),          # transform the data from a PIL image to a tensor
        train=True,                               # is this training data?
        download=True                             # should i download it if it's not already here?
    )

    # GET THE TEST DATASET
    test_data = datasets.MNIST(root='MNIST-data',
        transform=transforms.ToTensor(),
        train=False,
    )

    train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])    # split into 50K training & 10K validation
    
    return train_data, val_data, test_data


def get_dataloaders(batch_size=16):

    train_data, val_data, test_data = get_splits()

    # MAKE TRAINING DATALOADER
    train_loader = torch.utils.data.DataLoader( # create a data loader
        train_data, # what dataset should it sample from?
        shuffle=True, # should it shuffle the examples?
        batch_size=batch_size # how large should the batches that it samples be?
    )

    # MAKE VALIDATION DATALOADER
    val_loader = torch.utils.data.DataLoader(
        val_data,
        shuffle=True,
        batch_size=batch_size
    )

    # MAKE TEST DATALOADER
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader